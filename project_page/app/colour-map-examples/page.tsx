export default function ColourMapExamples() {
    const root_path = process.env.NEXT_PUBLIC_BASE_PATH || "";
    const original = `${root_path}/apps_editing_original.png`;
    const colourMapOriginal = `${root_path}/apps_editing_seg_original.png`;
    const colourMapEdited = `${root_path}/apps_editing_seg_edited.png`;
    const controlnet = `${root_path}/apps_editing_controlnet_prompt.png`;
    const t2iadapter = `${root_path}/apps_editing_t2iadapter_prompt.png`;
    const controlnetpp = `${root_path}/apps_editing_controlnetpp_prompt.png`;
    const ours = `${root_path}/apps_editing_ours.png`;

    return (
        <main className="mx-auto max-w-5xl py-10 px-4 sm:px-6 lg:px-8 text-gray-800">
            <h1 className="text-2xl font-semibold mb-4">Colour Map Editing Examples</h1>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                <figure className="rounded-xl border p-2">
                    <img src={original} alt="Original image" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">Original Image</figcaption>
                </figure>
                <figure className="rounded-xl border p-2">
                    <img src={colourMapOriginal} alt="Original colour map" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">Original Colour Map</figcaption>
                </figure>
                <figure className="rounded-xl border p-2">
                    <img src={colourMapEdited} alt="Edited colour map" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">Edited Colour Map</figcaption>
                </figure>
                <figure className="rounded-xl border p-2">
                    <img src={controlnet} alt="ControlNet with prompt" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">ControlNet w/ Prompt</figcaption>
                </figure>
                <figure className="rounded-xl border p-2">
                    <img src={t2iadapter} alt="T2I Adapter with prompt" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">T2I Adapter w/ Prompt</figcaption>
                </figure>
                <figure className="rounded-xl border p-2">
                    <img src={controlnetpp} alt="ControlNet++ with prompt" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">ControlNet++ w/ Prompt</figcaption>
                </figure>
                <figure className="rounded-xl border p-2">
                    <img src={ours} alt="Our method output" className="w-full h-40 object-cover rounded-lg" />
                    <figcaption className="mt-2 text-center text-xs">Ours</figcaption>
                </figure>
            </div>
        </main>
    );
}
