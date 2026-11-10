// The picture parts People and Pets share (duplicated frontend components, row 4): a round avatar for a roster row,
// and the big photo at the top of a card, each with the name's initials (the transcript's rule, transcript/model.ts)
// until there's a photo.
import { Avatar, Indicator } from '@mantine/core';
import type { ReactNode } from 'react';
import { SafeImage } from '../../shared/SafeImage';
import { initials } from '../transcript/model';
import classes from './people.module.css';
import card from './profile/profile.module.css';

/** A person's or pet's photo as a round avatar, or their initials while there's none; a green dot while they're here
 *  now (people only). */
export function EntityAvatar({
  name,
  photo,
  here = false,
  size = 36,
}: {
  name: string;
  photo: string | null | undefined;
  here?: boolean;
  size?: number;
}) {
  return (
    <Indicator
      className={classes.avatar}
      disabled={!here}
      position="bottom-end"
      offset={3}
      size={11}
      color="var(--lib-ok)"
      withBorder
      zIndex={1}
      aria-hidden
    >
      <Avatar src={photo || null} size={size} radius="xl" alt="" aria-hidden>
        {initials(name)}
      </Avatar>
    </Indicator>
  );
}

/** The big photo at the top of a card, or the initials (named for screen readers as having no photo yet). */
export function MainPhoto({ url, name }: { url: string | null | undefined; name: string }) {
  return url ? (
    <SafeImage className={card.mainPhoto} src={url} alt={name} width={152} height={152} />
  ) : (
    <div className={`${card.mainPhoto} ${card.initials}`} role="img" aria-label={`${name}, no photo yet`}>
      {initials(name)}
    </div>
  );
}

/** One photo on a card: its key, picture, and alt text, then its ⋯ menu (in the photo's top right corner, shown on
 *  hover; the menu's button takes card.tileMenu for that) and a caption under it, when it has them. */
export interface CardPhoto {
  key: string;
  src: string;
  alt: string;
  title?: string;
  menu?: ReactNode;
  caption?: ReactNode;
}

/** A card's photos, three square tiles a row: a person's clearest shots, a pet's recent sightings. */
export function PhotoGrid({ photos }: { photos: readonly CardPhoto[] }) {
  return (
    <ul className={card.photos}>
      {photos.map(ph => (
        <li key={ph.key} className={card.photoTile}>
          <SafeImage src={ph.src} alt={ph.alt} title={ph.title} loading="lazy" retries={2} />
          {ph.menu && <div className={card.tileCorner}>{ph.menu}</div>}
          {ph.caption}
        </li>
      ))}
    </ul>
  );
}
