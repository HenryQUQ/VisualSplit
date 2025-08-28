import fs from 'fs';
import path from 'path';

type ExampleItem = {
  label: string;
  src: string;
};

function listGroups(baseFsDir: string) {
  if (!fs.existsSync(baseFsDir)) return { groups: [] as string[], rootFiles: [] as string[] };
  const entries = fs.readdirSync(baseFsDir, { withFileTypes: true });
  const groups = entries.filter(e => e.isDirectory()).map(e => e.name).sort();
  const rootFiles = entries.filter(e => e.isFile()).map(e => e.name);
  return { groups, rootFiles };
}

function isImageFile(name: string) {
  return /\.(png|jpg|jpeg|webp|gif)$/i.test(name);
}

function pickByPattern(files: string[], patterns: RegExp[]): string | undefined {
  for (const re of patterns) {
    const hit = files.find(f => re.test(f));
    if (hit) return hit;
  }
  return undefined;
}

export default function ColourMapExamples() {
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
  const fsBase = path.join(process.cwd(), 'public', 'colour-map');
  const { groups, rootFiles } = listGroups(fsBase);

  const groupNames = groups.length > 0 ? groups : (rootFiles.some(isImageFile) ? ['.'] : []);

  const sections = groupNames.map(group => {
    const fsDir = group === '.' ? fsBase : path.join(fsBase, group);
    const webDir = group === '.' ? `${basePath}/colour-map` : `${basePath}/colour-map/${group}`;
    const files = fs.existsSync(fsDir) ? fs.readdirSync(fsDir).filter(isImageFile) : [];

    const items: ExampleItem[] = [];

    const original = pickByPattern(files, [
      /(^|[_\-])original(?!.*seg)/i,
    ]);
    const colourMapOriginal = pickByPattern(files, [
      // seg-original, segment_original, original_segmented, colour map original
      /(^|[_\-])(seg|segment|segmented)[_\-]?(orig|original)/i,
      /(^|[_\-])(orig|original)[_\-]?(seg|segment|segmented)/i,
      /(colour|color).*(orig|original)/i,
    ]);
    const colourMapEdited = pickByPattern(files, [
      // seg-edited, edited_segmented, colour map edited
      /(^|[_\-])(seg|segment|segmented)[_\-]?(edit|edited)/i,
      /(^|[_\-])(edit|edited)[_\-]?(seg|segment|segmented)/i,
      /(colour|color).*(edit|edited)/i,
    ]);
    // ControlNet++ (prefer matching this before plain ControlNet)
    const controlnetpp = pickByPattern(files, [
      /(controlnet(?:pp|\+\+|[_\-]?plus[_\-]?plus)).*(prompt|caption)/i,
    ]);
    const controlnet = pickByPattern(files, [
      // Exclude ControlNet++ variants implicitly by putting after controlnetpp
      /(controlnet).*(prompt|caption)/i,
    ]);
    const t2iadapter = pickByPattern(files, [
      /(t2i|adapter).*(prompt|caption)/i,
    ]);
    const ours = pickByPattern(files, [
      /(ours|visualsplit)/i,
    ]);

    if (original) items.push({ label: 'Original Image', src: `${webDir}/${original}` });
    if (colourMapOriginal) items.push({ label: 'Original Colour Map', src: `${webDir}/${colourMapOriginal}` });
    if (colourMapEdited) items.push({ label: 'Edited Colour Map', src: `${webDir}/${colourMapEdited}` });
    if (controlnet) items.push({ label: 'ControlNet (Prompt/Caption)', src: `${webDir}/${controlnet}` });
    if (t2iadapter) items.push({ label: 'T2I Adapter (Prompt/Caption)', src: `${webDir}/${t2iadapter}` });
    if (controlnetpp) items.push({ label: 'ControlNet++ (Prompt/Caption)', src: `${webDir}/${controlnetpp}` });
    if (ours) items.push({ label: 'Ours', src: `${webDir}/${ours}` });

    return { group, items };
  }).filter(section => section.items.length > 0);

  return (
    <main className="mx-auto max-w-5xl py-10 px-4 sm:px-6 lg:px-8 text-gray-800">
      <h1 className="text-2xl font-semibold mb-4">Colour Map Editing Examples</h1>
      {sections.length === 0 ? (
        <p className="text-sm text-gray-500">No examples found under <code>public/colour-map</code>.</p>
      ) : (
        sections.map(({ group, items }) => (
          <section key={group} className="mb-8">
            {group !== '.' && (
              <h2 className="text-lg font-medium mb-3">{group}</h2>
            )}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
              {items.map(({ label, src }) => (
                <figure key={label} className="rounded-xl border p-2">
                  <img src={src} alt={label} className="w-full object-cover rounded-lg" />
                  <figcaption className="mt-2 text-center text-xs">{label}</figcaption>
                </figure>
              ))}
            </div>
          </section>
        ))
      )}
    </main>
  );
}
