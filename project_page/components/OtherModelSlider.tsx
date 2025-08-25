'use client';

import { useState } from 'react';

interface ModelOption {
  label: string;
  withPrompt: string;
  withoutPrompt: string;
}

interface OtherModelSliderProps {
  options: ModelOption[];
  ours: string;
  gt: string;
}

export default function OtherModelSlider({ options, ours, gt }: OtherModelSliderProps) {
  const [index, setIndex] = useState(0);
  const current = options[index];
  return (
    <div className="rounded-xl border p-2 mt-4">
      <h5 className="font-medium mb-2 text-center">Other Models</h5>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        <figure>
          <img
            src={current.withoutPrompt}
            alt={`${current.label} w/o prompt`}
            className="w-full object-cover rounded-lg"
          />
          <figcaption className="mt-1 text-center text-xs">w/o prompt</figcaption>
        </figure>
        <figure>
          <img
            src={current.withPrompt}
            alt={`${current.label} w/ prompt`}
            className="w-full object-cover rounded-lg"
          />
          <figcaption className="mt-1 text-center text-xs">w/ prompt</figcaption>
        </figure>
        <figure>
          <img
            src={ours}
            alt="Ours"
            className="w-full object-cover rounded-lg"
          />
          <figcaption className="mt-1 text-center text-xs">Ours</figcaption>
        </figure>
        <figure>
          <img
            src={gt}
            alt="Ground truth"
            className="w-full object-cover rounded-lg"
          />
          <figcaption className="mt-1 text-center text-xs">Original Image</figcaption>
        </figure>
      </div>
      <input
        type="range"
        min={0}
        max={options.length - 1}
        step={1}
        value={index}
        onChange={(e) => setIndex(parseInt(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-xs mt-1">
        {options.map((opt, i) => (
          <span key={opt.label} className={i === index ? 'font-semibold' : ''}>
            {opt.label}
          </span>
        ))}
      </div>
    </div>
  );
}

