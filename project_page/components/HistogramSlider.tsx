'use client';

import {useState} from 'react';

interface HistogramOption {
    label: string;
    histSrc: string;
    oursSrc: string;
}

interface HistogramSliderProps {
    before: string;
    options: HistogramOption[];
}

export default function HistogramSlider({before, options}: HistogramSliderProps) {
    const [index, setIndex] = useState(0);
    return (
        <div className="rounded-xl border p-2">
            <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="rounded-xl border">
                    <figure className="p-2">
                        <img src={before} alt="Gray-level histogram (Before)"
                             className="w-full object-cover rounded-lg"/>
                        <figcaption className="mt-2 text-center text-xs">Gray-Level Histogram (Before)</figcaption>
                    </figure>
                    <figure className="p-2">
                        <img src={options[index].histSrc} alt="Gray-level histogram (After)"
                             className="w-full object-cover rounded-lg"/>
                        <figcaption className="mt-2 text-center text-xs">Gray-Level Histogram (After)</figcaption>
                    </figure>
                </div>
                <figure className="rounded-xl border p-2">
                    <img src={options[index].oursSrc} alt="Ours" className="w-full object-cover rounded-lg"/>
                    <figcaption className="mt-2 text-center text-xs">Ours</figcaption>
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

