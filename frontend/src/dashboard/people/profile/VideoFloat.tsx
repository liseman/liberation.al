// A People or pet card's recent video, opened bigger: the shared floating clip window (clip/ClipFloat.tsx), named
// for whose card it's on, with the card's still as its poster and Review's person filter on Open in Review.
import { ClipFloat } from '../../clip/ClipFloat';
import type { LoadSegments } from '../../clip/clip';
import type { MakeVideo } from '../../review/Player';
import type { CardVideo } from '../api';

export interface VideoFloatProps {
  video: CardVideo;
  name: string;
  person?: string | null; // Review's person filter for "open in Review": defaults to name, null for none (a pet)
  origin: HTMLElement;
  onClose: () => void;
  makeVideo?: MakeVideo;
  loadSegments?: LoadSegments;
}

export function VideoFloat({ video, name, person, origin, onClose, makeVideo, loadSegments }: VideoFloatProps) {
  return (
    <ClipFloat
      cam={video.cam}
      start={video.start}
      end={video.end}
      label={`Video of ${name}`}
      origin={origin}
      onClose={() => onClose()}
      poster={video.thumb}
      person={person === undefined ? name : (person ?? undefined)}
      makeVideo={makeVideo}
      loadSegments={loadSegments}
    />
  );
}
