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
            <div className="grid grid-cols-2 gap-3 mb-4 items-stretch">
                <div className="rounded-xl border flex flex-col gap-3">
                    <figure className="p-3 flex flex-1 flex-col items-center justify-center text-center">
                        <img src={before} alt="Gray-level histogram (Before)"
                             className="rounded-lg max-h-48 w-full object-contain"/>
                        <figcaption className="mt-2 text-xs">Gray-Level Histogram (Before)</figcaption>
                    </figure>
                    <figure className="p-3 flex flex-1 flex-col items-center justify-center text-center">
                        <img src={options[index].histSrc} alt="Gray-level histogram (After)"
                             className="rounded-lg max-h-48 w-full object-contain"/>
                        <figcaption className="mt-2 text-xs">Gray-Level Histogram (After)</figcaption>
                    </figure>
                </div>
                <figure className="rounded-xl border p-3 flex flex-col items-center justify-center text-center">
                    <img src={options[index].oursSrc} alt="Ours"
                         className="rounded-lg max-h-60 w-full object-contain"/>
                    <figcaption className="mt-2 text-xs">Ours</figcaption>
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
            <div className="relative mt-2 h-6 text-xs">
                {options.map((opt, i) => {
                    const percent = options.length > 1 ? (i / (options.length - 1)) * 100 : 0;
                    const translate =
                        i === 0
                            ? "translateX(0%)"
                            : i === options.length - 1
                                ? "translateX(-100%)"
                                : "translateX(-50%)";
                    const alignment =
                        i === 0 ? "text-left" : i === options.length - 1 ? "text-right" : "text-center";
                    return (
                        <span
                            key={opt.label}
                            className={`${i === index ? "font-semibold" : ""} ${alignment} absolute top-0 whitespace-nowrap`}
                            style={{ left: `${percent}%`, transform: translate }}
                        >
                            {opt.label}
                        </span>
                    );
                })}
            </div>
        </div>
    );
}
