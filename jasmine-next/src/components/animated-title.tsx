'use client';

import { useEffect, useRef } from 'react';
import { createTimeline, stagger } from 'animejs';

const LINES = ['JASMINE', 'early autism diagnosis', 'through movement'];

export function AnimatedTitle() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const lineEls = el.querySelectorAll<HTMLElement>('.type-line');

    lineEls.forEach((line, idx) => {
      const text = line.textContent || '';
      line.innerHTML = text
        .split('')
        .map((char) =>
          char === ' '
            ? ' '
            : `<span class="letter">${char}</span>`,
        )
        .join('');

      if (idx === 0) {
        line.querySelectorAll<HTMLElement>('.letter').forEach((span) => {
          span.style.backgroundImage = 'var(--gradient-primary)';
          span.style.webkitBackgroundClip = 'text';
          span.style.backgroundClip = 'text';
          span.style.color = 'transparent';
        });
      }
    });

    const line0 = lineEls[0]?.querySelectorAll<HTMLElement>('.letter');
    const line1 = lineEls[1]?.querySelectorAll<HTMLElement>('.letter');
    const line2 = lineEls[2]?.querySelectorAll<HTMLElement>('.letter');

    // Set initial hidden state
    el.querySelectorAll<HTMLElement>('.letter').forEach(
      (s) => (s.style.opacity = '0'),
    );

    const tl = createTimeline({ autoplay: false });

    if (line0?.length) {
      tl.add(line0, {
        opacity: [0, 1],
        ease: 'outExpo',
        duration: 400,
        delay: stagger(50),
      });
    }

    if (line1?.length) {
      tl.add(line1, {
        opacity: [0, 1],
        ease: 'outExpo',
        duration: 400,
        delay: stagger(50),
      });
    }

    if (line2?.length) {
      tl.add(line2, {
        opacity: [0, 1],
        ease: 'outExpo',
        duration: 400,
        delay: stagger(50),
      });
    }

    const timeout = setTimeout(() => tl.play(), 900);
    return () => {
      clearTimeout(timeout);
      tl.pause();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-[1.1] tracking-tight space-y-2"
    >
      {LINES.map((line, i) => (
        <div
          key={i}
          className={`type-line ${i === 0 ? 'text-6xl sm:text-7xl md:text-8xl lg:text-9xl' : ''}`}
          style={{ color: 'var(--text-primary)' }}
        >
          {line}
        </div>
      ))}
    </div>
  );
}
