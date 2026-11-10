// Stands in for `next/image`, which Pascal's editor imports: a plain <img>, without Next's own props.
import type { ImgHTMLAttributes } from 'react';

type Props = Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'> & {
  src: string | { src: string };
  fill?: boolean;
  priority?: boolean;
  unoptimized?: boolean;
  quality?: number;
  placeholder?: string;
  blurDataURL?: string;
};

export default function Image({
  src,
  alt,
  fill,
  priority,
  unoptimized,
  quality,
  placeholder,
  blurDataURL,
  ...rest
}: Props) {
  void [fill, priority, unoptimized, quality, placeholder, blurDataURL];
  return <img src={typeof src === 'string' ? src : src.src} alt={alt ?? ''} {...rest} />;
}
