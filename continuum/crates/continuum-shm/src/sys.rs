//! POSIX shared memory: `shm_open`, a 16 KB aligned `mmap`, and the monotonic clock. Linux and macOS.

use std::ffi::CString;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd};
use std::ptr::NonNull;

use continuum_core::PAGE_ALIGN;

use crate::Error;

/// macOS caps shared memory names at 31 bytes (`PSHMNAMLEN`), slash included. Linux allows 255, but we hold every
/// platform to the macOS limit so a name that works here works on the deploy target.
pub const MAX_NAME_LEN: usize = 31;

/// Checks a shared memory name: a leading slash, no other slash, no NUL, and at most [`MAX_NAME_LEN`] bytes.
pub fn check_name(name: &str) -> Result<CString, Error> {
    let ok = name.len() > 1
        && name.len() <= MAX_NAME_LEN
        && name.starts_with('/')
        && !name[1..].contains('/')
        && !name.contains('\0');
    if !ok {
        return Err(Error::BadName(name.to_owned()));
    }
    Ok(CString::new(name).expect("checked for NUL above"))
}

/// A shared memory object mapped read-write at a [`PAGE_ALIGN`] aligned address. Unmapped on drop.
pub struct Mapping {
    base: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is plain memory shared between processes; every access to it goes through atomics or through
// the slot and ring protocols, which don't depend on which thread holds the `Mapping`.
unsafe impl Send for Mapping {}
// SAFETY: as above.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Creates a new shared memory object of `len` bytes (a multiple of [`PAGE_ALIGN`]) and maps it. Fails if the
    /// name exists. The memory starts zeroed.
    pub fn create(name: &str, len: usize) -> Result<Self, Error> {
        assert!(len > 0 && len.is_multiple_of(PAGE_ALIGN), "arena length must be a positive multiple of PAGE_ALIGN");
        let cname = check_name(name)?;
        let off_len = libc::off_t::try_from(len).map_err(|_| Error::TooLarge)?;
        // SAFETY: cname is a valid NUL-terminated string.
        let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_EXCL | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        let fd = Fd(fd);
        // SAFETY: fd is an open shared memory object. macOS allows one ftruncate per object, which this is.
        if unsafe { libc::ftruncate(fd.0, off_len) } != 0 {
            let err = io::Error::last_os_error();
            unlink(name).ok();
            return Err(Error::Io(err));
        }
        map_aligned(fd.0, len).inspect_err(|_| {
            unlink(name).ok();
        })
    }

    /// Maps an existing shared memory object, at its full size.
    pub fn open(name: &str) -> Result<Self, Error> {
        let cname = check_name(name)?;
        // SAFETY: cname is a valid NUL-terminated string.
        let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0) };
        if fd < 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        let fd = Fd(fd);
        // SAFETY: fd is open until the end of this function.
        Self::from_fd(unsafe { BorrowedFd::borrow_raw(fd.0) })
    }

    /// Maps a shared memory object from an open file descriptor (one received from the broker), at its full size.
    /// The mapping holds its own reference, so the caller may close `fd` afterward.
    pub fn from_fd(fd: BorrowedFd<'_>) -> Result<Self, Error> {
        // SAFETY: an all-zero stat is a valid value to overwrite.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: fd is open and st is a valid out pointer.
        if unsafe { libc::fstat(fd.as_raw_fd(), &raw mut st) } != 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        let len = usize::try_from(st.st_size).map_err(|_| Error::TooLarge)?;
        if len == 0 || !len.is_multiple_of(PAGE_ALIGN) {
            // Not sized yet (the creator is between shm_open and ftruncate), or not one of ours.
            return Err(Error::NotReady);
        }
        map_aligned(fd.as_raw_fd(), len)
    }

    /// The first byte of the mapping, aligned to [`PAGE_ALIGN`].
    pub fn base(&self) -> *mut u8 {
        self.base.as_ptr()
    }

    /// The mapping's size in bytes.
    pub fn size(&self) -> usize {
        self.len
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: base and len are exactly what map_aligned mapped, and nothing borrows the mapping past its owner.
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.len) };
    }
}

/// Removes a shared memory name. Mappings that are open stay valid until they're unmapped.
pub fn unlink(name: &str) -> Result<(), Error> {
    let cname = check_name(name)?;
    // SAFETY: cname is a valid NUL-terminated string.
    if unsafe { libc::shm_unlink(cname.as_ptr()) } != 0 {
        return Err(Error::Io(io::Error::last_os_error()));
    }
    Ok(())
}

struct Fd(libc::c_int);

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: we own the descriptor. The mapping outlives it, as POSIX allows.
        unsafe { libc::close(self.0) };
    }
}

/// Maps `len` bytes of `fd` at a [`PAGE_ALIGN`] aligned address.
///
/// mmap only promises the system page size (4 KB on x86 Linux), so this reserves `len + PAGE_ALIGN` bytes of address
/// space, maps the object over the first 16 KB boundary inside it, and returns the unused ends. On Apple Silicon the
/// reservation is already aligned and nothing is trimmed.
fn map_aligned(fd: libc::c_int, len: usize) -> Result<Mapping, Error> {
    let reserve = len.checked_add(PAGE_ALIGN).ok_or(Error::TooLarge)?;
    // SAFETY: an anonymous PROT_NONE reservation; no memory is touched.
    let r = unsafe {
        libc::mmap(std::ptr::null_mut(), reserve, libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0)
    };
    if r == libc::MAP_FAILED {
        return Err(Error::Io(io::Error::last_os_error()));
    }
    let start = r as usize;
    let aligned = start.next_multiple_of(PAGE_ALIGN);
    // SAFETY: [aligned, aligned + len) lies inside the reservation we own, so MAP_FIXED replaces only our pages.
    let p = unsafe {
        libc::mmap(
            aligned as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_FIXED,
            fd,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        let err = io::Error::last_os_error();
        // SAFETY: the reservation is ours and unused.
        unsafe { libc::munmap(r, reserve) };
        return Err(Error::Io(err));
    }
    // SAFETY: the head and tail are the unused parts of our reservation, and both are system page aligned (start is,
    // aligned and len are multiples of 16 KB, and 16 KB is a multiple of every supported system page size).
    unsafe {
        if aligned > start {
            libc::munmap(r, aligned - start);
        }
        let end = aligned + len;
        if start + reserve > end {
            libc::munmap(end as *mut libc::c_void, start + reserve - end);
        }
    }
    let base = NonNull::new(aligned as *mut u8).expect("mmap never returns null on success");
    Ok(Mapping { base, len })
}

/// Nanoseconds on the system-wide monotonic clock, which every process on the machine shares (lease deadlines are
/// compared across processes, so `Instant` won't do).
pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: ts is a valid out pointer; CLOCK_MONOTONIC exists on Linux and on macOS 10.12 and later.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
    u64::try_from(ts.tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_macos_rules() {
        assert!(check_name("/cnt").is_ok());
        assert!(check_name(&format!("/{}", "a".repeat(30))).is_ok());
        assert!(check_name(&format!("/{}", "a".repeat(31))).is_err());
        assert!(check_name("cnt").is_err());
        assert!(check_name("/").is_err());
        assert!(check_name("/a/b").is_err());
        assert!(check_name("/a\0b").is_err());
    }
}
