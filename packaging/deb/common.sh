# The start of every maintainer script (tools/apt_package.py puts it there): what they share.
#
# The package's own files are root's, under /usr. Everything else is each user's: the environment pack-setup built
# (~/.local/lib/liberational), the settings (~/.config/liberational), and the data (~/.local/share/liberational).
# A user who moved the last two with XDG_CONFIG_HOME or XDG_DATA_HOME is not found by these scripts, which is the
# safe side: nothing of theirs is touched.
#
# A home is its user's to arrange, links included, so root only looks for names there (users_with). Everything that
# reads, writes, or deletes inside a home runs as that user (as_user).
set -e

ENV_DIR=.local/lib/liberational
CONFIG_DIR=.config/liberational
DATA_DIR=.local/share/liberational
AUTOSTART=.config/autostart/liberational.desktop
QUIT_WAIT_S=300

# users_with <path under a home> ...: "name:home" for each person's account that has any of them.
users_with() {
  getent passwd | while IFS=: read -r name _ uid _ _ home _; do
    [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] || continue
    for path in "$@"; do
      if [ -e "$home/$path" ]; then
        echo "$name:$home"
        break
      fi
    done
  done
}

# The running applications: every process that is a packaged environment's pack-app or `pack-supervisor run`.
app_pids() {
  for dir in /proc/[0-9]*; do
    [ -r "$dir/cmdline" ] || continue
    line="$(tr '\0' ' ' < "$dir/cmdline" 2> /dev/null)" || continue
    case "$line" in
      *"/$ENV_DIR/liberational-pack/bin/pack-app "* | *"/$ENV_DIR/liberational-pack/bin/pack-supervisor run "*)
        echo "${dir#/proc/}"
        ;;
    esac
  done
}

# Have every running application drain and quit, and wait until it has. SIGTERM is its Quit: the supervisor stops
# the readers, then capture, lets the tier worker flush, and stops the broker last. One that is still there after
# QUIT_WAIT_S is killed, and its services with it, so the environment is never replaced under a running program.
quit_applications() {
  pids="$(app_pids)"
  [ -n "$pids" ] || return 0
  echo "Liberational is running: asking it to finish what it recorded and quit"
  # shellcheck disable=SC2086
  kill -TERM $pids 2> /dev/null || true
  waited=0
  while [ -n "$(app_pids)" ] && [ "$waited" -lt "$QUIT_WAIT_S" ]; do
    sleep 1
    waited=$((waited + 1))
  done
  pids="$(app_pids)"
  if [ -n "$pids" ]; then
    echo "Liberational did not quit in $QUIT_WAIT_S seconds: stopping it now. What it had not yet stored is lost."
    # shellcheck disable=SC2086
    kill -KILL $pids 2> /dev/null || true
    sleep 2
  else
    echo "Liberational quit after $waited seconds"
  fi
}

# as_user <name> <home> <command> ...: the command as that user, with a plain environment of their own, and its
# exit status. What it runs may be the user's own code (their environment's pack-setup), so it gets none of root's
# terminal: a session of its own (setsid), no input, and its output through a pipe. A process that shared the
# terminal could type into it as root.
as_user() {
  user="$1" home="$2"
  shift 2
  status="$({
    {
      setsid -w runuser -u "$user" -- env -i HOME="$home" USER="$user" LOGNAME="$user" \
        PATH=/usr/local/bin:/usr/bin:/bin LANG="${LANG:-C.UTF-8}" ${http_proxy:+http_proxy="$http_proxy"} \
        ${https_proxy:+https_proxy="$https_proxy"} ${no_proxy:+no_proxy="$no_proxy"} "$@" < /dev/null 2>&1 3>&- 4>&-
      echo "$?" >&3
    } | cat >&4
  } 3>&1)" 4>&1
  return "${status:-1}"
}
