'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function RestorationTable() {
  const [showTable, setShowTable] = useState(false);

  if (!showTable) {
    return (
      <div className="mt-4">
        <button
          className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md"
          onClick={() => setShowTable(true)}
        >
          Show image restoration results
        </button>
      </div>
    );
  }

  return (
    <div className="mt-4">
      <p className="text-sm mb-2 font-medium">Image restoration results.</p>
      <div className="overflow-x-auto rounded-2xl border">
        <table className="w-full text-sm border-collapse">
          <thead className="bg-gray-50">
            <tr>
              <th className="border px-3 py-2 text-left font-semibold">Method</th>
              <th className="border px-3 py-2 text-center font-semibold">Prompt</th>
              <th className="border px-3 py-2 text-center font-semibold">PSNR</th>
              <th className="border px-3 py-2 text-center font-semibold">SSIM</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td rowSpan={2} className="border px-3 py-2 align-top">ControlNet</td>
              <td className="border px-3 py-2 text-center">w/o</td>
              <td className="border px-3 py-2 text-center">17.34</td>
              <td className="border px-3 py-2 text-center">0.6374</td>
            </tr>
            <tr>
              <td className="border px-3 py-2 text-center">w/</td>
              <td className="border px-3 py-2 text-center">16.52</td>
              <td className="border px-3 py-2 text-center">0.6051</td>
            </tr>
            <tr>
              <td rowSpan={2} className="border px-3 py-2 align-top">T2I Adapter</td>
              <td className="border px-3 py-2 text-center">w/o</td>
              <td className="border px-3 py-2 text-center">17.69</td>
              <td className="border px-3 py-2 text-center">0.5421</td>
            </tr>
            <tr>
              <td className="border px-3 py-2 text-center">w/</td>
              <td className="border px-3 py-2 text-center">17.30</td>
              <td className="border px-3 py-2 text-center">0.5459</td>
            </tr>
            <tr>
              <td rowSpan={2} className="border px-3 py-2 align-top">ControlNet++</td>
              <td className="border px-3 py-2 text-center">w/o</td>
              <td className="border px-3 py-2 text-center">19.94</td>
              <td className="border px-3 py-2 text-center">0.6549</td>
            </tr>
            <tr>
              <td className="border px-3 py-2 text-center">w/</td>
              <td className="border px-3 py-2 text-center">19.50</td>
              <td className="border px-3 py-2 text-center">0.6399</td>
            </tr>
            <tr className="bg-gray-50">
              <td className="border px-3 py-2 font-semibold">VisualSplit (Ours)</td>
              <td className="border px-3 py-2 text-center">w/o</td>
              <td className="border px-3 py-2 text-center font-semibold">26.56</td>
              <td className="border px-3 py-2 text-center font-semibold">0.8664</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-[11px] mt-2 text-gray-500">Prompts are not required for our method.</p>
      <div className="text-right mt-4">
        <Link href="/restoration-examples">
          <button className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md">
            View more examples
          </button>
        </Link>
      </div>
    </div>
  );
}

