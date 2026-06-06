'use client';

import { useRef, useEffect } from 'react';

type LimbGroup = 'head' | 'torso' | 'leftArm' | 'rightArm' | 'leftLeg' | 'rightLeg' | 'foot';

const LIMB_GROUPS: Record<LimbGroup, { connections: [number, number][]; color: string; darkColor: string; label: string }> = {
  head: {
    connections: [[0, 1], [0, 15], [0, 16], [15, 17], [16, 18]],
    color: '#e8a838', darkColor: '#f0c060',
    label: 'Head',
  },
  torso: {
    connections: [[1, 8]],
    color: '#74b3ce', darkColor: '#8dd0e8',
    label: 'Torso',
  },
  leftArm: {
    connections: [[1, 5], [5, 6], [6, 7]],
    color: '#4caf50', darkColor: '#66bb6a',
    label: 'Left Arm',
  },
  rightArm: {
    connections: [[1, 2], [2, 3], [3, 4]],
    color: '#ef5350', darkColor: '#e57373',
    label: 'Right Arm',
  },
  leftLeg: {
    connections: [[8, 12], [12, 13], [13, 14]],
    color: '#ab47bc', darkColor: '#ce93d8',
    label: 'Left Leg',
  },
  rightLeg: {
    connections: [[8, 9], [9, 10], [10, 11]],
    color: '#ff7043', darkColor: '#ff8a65',
    label: 'Right Leg',
  },
  foot: {
    connections: [[11, 24], [24, 23], [23, 22], [14, 21], [21, 20], [20, 19]],
    color: '#78909c', darkColor: '#90a4ae',
    label: 'Feet',
  },
};

interface PoseViewerProps {
  keypoints: number[][];
  width?: number;
  height?: number;
  showLabels?: boolean;
  showLegend?: boolean;
}

export function PoseViewer({ keypoints, width = 280, height = 380, showLabels = false, showLegend = false }: PoseViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !Array.isArray(keypoints) || keypoints.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const isDark = document.documentElement.classList.contains('dark');
    const bgColor = isDark ? '#1a1a2e' : '#f0f4f8';

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

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [, group] of Object.entries(LIMB_GROUPS)) {
      const color = isDark ? group.darkColor : group.color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      for (const [i, j] of group.connections) {
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
    }

    // Joints
    points.forEach((p, idx) => {
      if (p.conf > 0.1) {
        const radius = p.conf > 0.5 ? 4 : 2.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isDark ? '#ffffff' : '#1a1a2e';
        ctx.fill();
        ctx.strokeStyle = isDark ? '#cccccc' : '#555555';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });

    // Joint labels
    if (showLabels) {
      const labelPositions: Record<number, string> = {
        0: 'Nose', 1: 'Neck', 2: 'RShou', 3: 'RElb', 4: 'RWrist',
        5: 'LShou', 6: 'LElb', 7: 'LWrist', 8: 'Hip',
        9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle',
      };
      ctx.font = 'bold 8px sans-serif';
      ctx.textAlign = 'center';
      points.forEach((p, idx) => {
        if (p.conf > 0.5 && idx in labelPositions) {
          ctx.fillStyle = isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)';
          ctx.fillText(labelPositions[idx], p.x, p.y - 7);
        }
      });
    }

    // Legend
    if (showLegend) {
      const legendX = 8;
      const legendY = height - 8;
      ctx.font = '8px sans-serif';
      let offsetX = 0;
      for (const [, group] of Object.entries(LIMB_GROUPS)) {
        const color = isDark ? group.darkColor : group.color;
        const txt = group.label;
        const txtW = ctx.measureText(txt).width + 14;
        const lx = legendX + offsetX;
        if (lx + txtW > width - 8) break;

        ctx.fillStyle = color;
        ctx.fillRect(lx, legendY - 7, 8, 8);
        ctx.fillStyle = isDark ? '#cccccc' : '#444444';
        ctx.fillText(txt, lx + 11, legendY);
        offsetX += txtW + 6;
      }
    }
  }, [keypoints, width, height, showLabels, showLegend]);

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
      style={{ borderColor: 'var(--border)', width, height }}
    />
  );
}
