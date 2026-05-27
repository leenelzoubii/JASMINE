'use client';

import { useRef, useEffect } from 'react';

const BODY25_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
  [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14],
  [0, 15], [0, 16], [15, 17], [16, 18],
  [11, 24], [24, 23], [23, 22], [14, 21], [21, 20], [20, 19],
];

const KEYPOINT_LABELS: Record<number, string> = {
  0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
  5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip',
  9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle',
};

interface PoseViewerProps {
  keypoints: number[][];
  width?: number;
  height?: number;
}

export function PoseViewer({ keypoints, width = 280, height = 380 }: PoseViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !Array.isArray(keypoints) || keypoints.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    const isDark = document.documentElement.classList.contains('dark');
    const bgColor = isDark ? '#1a1a2e' : '#f0f4f8';
    const skeletonColor = isDark ? '#74b3ce' : '#3589a8';
    const jointColor = isDark ? '#d6f3f4' : '#004346';

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    const minX = Math.min(...keypoints.map(k => k[0]));
    const maxX = Math.max(...keypoints.map(k => k[0]));
    const minY = Math.min(...keypoints.map(k => k[1]));
    const maxY = Math.max(...keypoints.map(k => k[1]));

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padding = 30;
    const scaleX = (width - padding * 2) / rangeX;
    const scaleY = (height - padding * 2) / rangeY;
    const scale = Math.min(scaleX, scaleY) * 0.85;

    const centerX = width / 2;
    const centerY = height / 2;

    const points = keypoints.map(kp => ({
      x: centerX + (kp[0] - (minX + maxX) / 2) * scale,
      y: centerY + (kp[1] - (minY + maxY) / 2) * scale,
      conf: kp[2],
    }));

    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = skeletonColor;

    for (const [i, j] of BODY25_CONNECTIONS) {
      if (i < points.length && j < points.length) {
        const p1 = points[i];
        const p2 = points[j];
        if (p1.conf > 0.1 && p2.conf > 0.1) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }

    points.forEach((p, idx) => {
      if (p.conf > 0.1) {
        const radius = p.conf > 0.5 ? 4 : 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = jointColor;
        ctx.fill();
        if (idx in KEYPOINT_LABELS) {
          ctx.fillStyle = isDark ? '#aaa' : '#666';
          ctx.font = '9px sans-serif';
          ctx.fillText(KEYPOINT_LABELS[idx], p.x + 5, p.y - 5);
        }
      }
    });
  }, [keypoints, width, height]);

  if (!keypoints || keypoints.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-xl" style={{ width, height, backgroundColor: 'var(--background-alt)' }}>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No pose data</p>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-xl border"
      style={{ borderColor: 'var(--border)' }}
    />
  );
}
