'use client';

import { useState } from 'react';

interface Option {
  label: string;
  src: string;
}

interface ImageSliderProps {
  title: string;
  description: string;
  beforeSrc: string;
  options: Option[];
}

export default function ImageSlider({ title, description, beforeSrc, options }: ImageSliderProps) {
  const [index, setIndex] = useState(0);
  return (
    <div className="rounded-2xl border p-4">
      <h4 className="font-semibold mb-2">{title}</h4>
      <p className="text-sm mb-3">{description}</p>
      <div className="grid grid-cols-2 gap-2 mb-4">
        <img
          src={beforeSrc}
          alt="Before"
          className="w-full object-cover rounded-lg"
        />
        <img
          src={options[index].src}
          alt={options[index].label}
          className="w-full object-cover rounded-lg"
        />
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
          <span
            key={opt.label}
            className={i === index ? 'font-semibold' : ''}
          >
            {opt.label}
          </span>
        ))}
      </div>
    </div>
  );
}

