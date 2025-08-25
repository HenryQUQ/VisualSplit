// This file is a Server Component. Avoid Client-only features such as
// `use client` directives or React hooks so that the page can be rendered
// entirely on the server.

// Metadata for the page is declared via the `metadata` export which is
// supported by the Next.js App Router for Server Components.
import ImageSlider from "@/components/ImageSlider";
import OtherModelSlider from "@/components/OtherModelSlider";
import HistogramSlider from "@/components/HistogramSlider";
import Link from "next/link";
import RestorationTable from "@/components/RestorationTable";

export const metadata = {
    title: "VisualSplit: Decoupled Visual Descriptors for Image Representation",
    description:
        "Project page for VisualSplit (BMVC 2025): image representation learned from decoupled, interpretable classical descriptors.",
};

/*
 * This page showcases the key ideas from the BMVC 2025 paper
 * "Exploring Image Representation with Decoupled Classical Visual
 * Descriptors" (VisualSplit). All text here is a concise, original
 * summary suitable for a public-facing project page.
 */

// Resolve the base path for static assets. In production on GitHub Pages,
// assets are served from the repository sub-path, while in development they
// live at the domain root.
const root_path = process.env.NEXT_PUBLIC_BASE_PATH || "";

// Static assets. Replace these with your own files under `public/`.
const hero_bg_path = `${root_path}/visualsplit_hero.jpg`;
const framework_path = `${root_path}/framework.png`;

// Author and affiliation data models.
interface Author {
    name: string;
    affiliations: number[];
    email?: string;
    link?: string;
}

interface Affiliation {
    id: number;
    name: string;
}

const affiliations: Affiliation[] = [
    {id: 1, name: "University of Birmingham, UK"},
    {id: 2, name: "University of Cambridge, UK"},
];

const authors: Author[] = [
    {
        name: "Chenyuan Qu",
        affiliations: [1],
        email: "cxq134@student.bham.ac.uk",
        link: "https://chenyuanqu.com/",
    },
    {
        name: "Hao Chen",
        affiliations: [1, 2],
        email: "hc666@cam.ac.uk",
        link: "https://h-chen.com/",
    },
    {
        name: "Jianbo Jiao",
        affiliations: [1],
        email: "j.jiao@bham.ac.uk",
        link: "https://jianbojiao.com/",
    },
];

// Image keys for the experiments section.
type View = "original" | "edges" | "segmentation" | "histogram" | "artist" | "output";

export default function VisualSplitPage() {
    // Demo images (place your generated examples under `public/`).
    const experimentImages: Record<View, string> = {
        original: `${root_path}/exp_original.png`,
        edges: `${root_path}/exp_edge.png`,
        segmentation: `${root_path}/exp_segmentation.png`,
        histogram: `${root_path}/exp_histogram.png`,
        artist: `${root_path}/exp_artist.png`,
        output: `${root_path}/exp_output.png`,
    };

    // Optional: additional static figures showing independent control/editing.
    const editImages = {
        illuminationBefore: `${root_path}/edit_illumination_before.png`,
        illuminationLow: `${root_path}/edit_illumination_low.png`,
        illuminationNormal: `${root_path}/edit_illumination_normal.png`,
        illuminationHigh: `${root_path}/edit_illumination_high.png`,
        colourBefore: `${root_path}/edit_colour_before.jpg`,
        colour025: `${root_path}/edit_colour_0.25.jpg`,
        colour075: `${root_path}/edit_colour_0.75.jpg`,
        colour100: `${root_path}/edit_colour_1.0.jpg`,
        colour125: `${root_path}/edit_colour_1.25.jpg`,
        colour175: `${root_path}/edit_colour_1.75.jpg`,
    };

    // Replace with your final links (PDF/code/models/dataset).
    const links = {
        paperPdf: `${root_path}/VisualSplit_BMVC2025.pdf`,
        suppPdf: `${root_path}/VisualSplit_supplementary.pdf`,
        arXiv: "#", // e.g., "/abs/xxxx.xxxxx"
        code: "#", // e.g., "https://github.com/your-repo/visualsplit"
        models: "https://huggingface.co/quchenyuan/VisualSplit",
        dataset: "#",
        poster: `${root_path}/VisualSplit_poster.pdf`,
        bibtexAnchor: "#bibtex",
    };
    const appAssets = {
        // Visual restoration descriptors and outputs
        restorationCond: `${root_path}/apps_restoration_condition.png`,
        restorationOurs: `${root_path}/apps_restoration_ours.png`,
        restorationGT: `${root_path}/apps_restoration_groundtruth.png`,

        // Other model results
        cnNoPrompt: `${root_path}/apps_restoration_controlnet_noprompt.png`,
        cnPrompt: `${root_path}/apps_restoration_controlnet_prompt.png`,
        t2iNoPrompt: `${root_path}/apps_restoration_t2iadapter_noprompt.png`,
        t2iPrompt: `${root_path}/apps_restoration_t2iadapter_prompt.png`,
        cnppNoPrompt: `${root_path}/apps_restoration_controlnetpp_noprompt.png`,
        cnppPrompt: `${root_path}/apps_restoration_controlnetpp_prompt.png`,

        // Descriptor-guided editing
        editingEdgeColour: `${root_path}/apps_editing_colour_edge.png`,
        editingHistColour: `${root_path}/apps_editing_colour_hist.png`,
        editingEdgeHist: `${root_path}/apps_editing_hist_edge.png`,
        editingOrig: `${root_path}/apps_editing_original.png`,
        editingColourOrig: `${root_path}/apps_editing_colour_original.png`,
        editingSegColourOrig: `${root_path}/apps_editing_colour_seg_original.png`,
        editingSegColourEdited: `${root_path}/apps_editing_colour_seg_edited.png`,
        editingSegHist: `${root_path}/apps_editing_hist_seg.png`,
        editingHistBefore: `${root_path}/apps_editing_hist_before.png`,
        editingHistAfter: `${root_path}/apps_editing_hist_after.png`,
        editingColourOut: `${root_path}/apps_editing_colour_output.png`,
        // Other model editing results
        editCnNoPrompt: `${root_path}/apps_editing_controlnet_noprompt.png`,
        editCnPrompt: `${root_path}/apps_editing_controlnet_prompt.png`,
        editT2iNoPrompt: `${root_path}/apps_editing_t2iadapter_noprompt.png`,
        editT2iPrompt: `${root_path}/apps_editing_t2iadapter_prompt.png`,
        editCnppNoPrompt: `${root_path}/apps_editing_controlnetpp_noprompt.png`,
        editCnppPrompt: `${root_path}/apps_editing_controlnetpp_prompt.png`,
    };

    const otherModelOptions = [
        {
            label: "ControlNet",
            withPrompt: appAssets.cnPrompt,
            withoutPrompt: appAssets.cnNoPrompt,
        },
        {
            label: "T2I-Adapter",
            withPrompt: appAssets.t2iPrompt,
            withoutPrompt: appAssets.t2iNoPrompt,
        },
        {
            label: "ControlNet++",
            withPrompt: appAssets.cnppPrompt,
            withoutPrompt: appAssets.cnppNoPrompt,
        },
    ];
    const editingModelOptions = [
        {
            label: "ControlNet",
            withPrompt: appAssets.editCnPrompt,
            withoutPrompt: appAssets.editCnNoPrompt,
        },
        {
            label: "T2I-Adapter",
            withPrompt: appAssets.editT2iPrompt,
            withoutPrompt: appAssets.editT2iNoPrompt,
        },
        {
            label: "ControlNet++",
            withPrompt: appAssets.editCnppPrompt,
            withoutPrompt: appAssets.editCnppNoPrompt,
        },
    ];
    const histogramLevels = ["-3", "-2", "-1", "0", "1", "2", "3", "equalization"];
    const histogramOptions = histogramLevels.map((level) => ({
        label: level,
        histSrc: `${root_path}/apps_editing_hist_after_${level}.png`,
        oursSrc: `${root_path}/apps_editing_output_${level}.png`,
    }));
    return (
        <>
            {/* Hero section */}
            <header id="home" className="relative w-full overflow-hidden">
                {/* Background image */}
                <img
                    src={hero_bg_path}
                    alt="Abstract illustration of edges, colour regions, and luminance"
                    className="w-full h-80 object-cover"
                />
                {/* Overlay with title and tagline */}
                <div
                    className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 text-white text-center px-4">
                    <h1 className="text-3xl sm:text-5xl font-bold mb-3">VisualSplit</h1>
                    <h2 className="text-lg sm:text-2xl max-w-3xl">
                        Exploring Image Representation with Decoupled Classical Visual
                        Descriptors
                    </h2>
                    <p className="mt-3 text-sm sm:text-base font-medium">
                        {authors.map((a, idx) => (
                            <span key={a.name}>
                                <a
                                    href={a.link}
                                    target="_blank"
                                    className="hover:underline"
                                >
                                    {a.name}
                                </a>
                                <sup>{a.affiliations.join(",")}</sup>
                                {idx < authors.length - 1 && " · "}
                            </span>
                        ))}
                    </p>
                    <p className="text-xs sm:text-sm">
                        {affiliations.map((aff, idx) => (
                            <span key={aff.id}>
                                <sup>{aff.id}</sup> {aff.name}
                                {idx < affiliations.length - 1 && " · "}
                            </span>
                        ))}
                    </p>
                    <p className="text-xs sm:text-sm mt-1">
                        <a
                            href="https://bmvc2025.bmva.org/"
                            className="underline hover:text-gray-300"
                        >
                            The 36th British Machine Vision Conference (BMVC 2025)
                        </a>
                    </p>
                    <div className="mt-4 flex flex-wrap justify-center gap-3">
                        <a
                            href={links.paperPdf}
                            target="_blank"
                            className="rounded-full bg-white/95 px-4 py-1.5 text-sm font-semibold text-gray-900 hover:bg-white"
                        >
                            📄 Paper (PDF)
                        </a>
                        <a
                            href={links.suppPdf}
                            target="_blank"
                            className="rounded-full bg-white/95 px-4 py-1.5 text-sm font-semibold text-gray-900 hover:bg-white"
                        >
                            📄 Supp (PDF)
                        </a>
                        <a
                            href={links.arXiv}
                            target="_blank"
                            aria-disabled="true"
                            className="rounded-full bg-white/80 px-4 py-1.5 text-sm font-semibold text-gray-900 opacity-50 cursor-not-allowed pointer-events-none"
                        >
                            📚 arXiv
                        </a>
                        <a
                            href={links.code}
                            target="_blank"
                            className="rounded-full bg-white/80 px-4 py-1.5 text-sm font-semibold text-gray-900 hover:bg-white"
                        >
                            💻 Code
                        </a>
                        <a
                            href={links.models}
                            target="_blank"
                            className="rounded-full bg-white/80 px-4 py-1.5 text-sm font-semibold text-gray-900 hover:bg-white"
                        >
                            🤗 Models
                        </a>
                        <a
                            href="#poster"
                            aria-disabled="true"
                            className="rounded-full bg-white/80 px-4 py-1.5 text-sm font-semibold text-gray-900 opacity-50 cursor-not-allowed pointer-events-none"
                        >
                            🖼️ Poster
                        </a>
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-5xl py-10 px-4 sm:px-6 lg:px-8 text-gray-800">
                {/* Introduction */}
                <section id="introduction" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">Introduction</h3>
                    <p className="mb-4">
                        Exploring and understanding efficient image representations is a long-standing challenge in
                        computer vision.
                        While deep learning has achieved remarkable progress across image understanding tasks, its
                        internal representations are often opaque, making it difficult to interpret how visual
                        information is processed. In contrast, classical visual descriptors (e.g. edge, colour, and
                        intensity distribution) have long been fundamental to image analysis and remain intuitively
                        understandable to humans.
                        Motivated by this gap, we ask a simple question: <i>can modern learning benefit from these
                        timeless cues?</i> In this paper we answer it with
                        {' '}<strong>VisualSplit</strong>, a framework that explicitly decomposes images into decoupled
                        classical descriptors,
                        treating each as an independent but complementary component of visual knowledge. Through a
                        reconstruction-driven pretraining scheme, {' '}<strong>VisualSplit</strong> learns to capture
                        the essence of each visual
                        descriptor while preserving their interpretability.
                        By explicitly decomposing visual attributes, our method inherently facilitates effective
                        attribute control in various advanced visual tasks, including image generation and editing,
                        extending beyond conventional classification and segmentation, suggesting the effectiveness of
                        this new learning approach towards visual understanding.
                    </p>
                </section>


                {/* Key Contributions */}
                <section id="contributions" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">Key Contributions</h3>
                    <ul className="list-disc list-inside space-y-2">
                        <li>
                            We revisit classical visual descriptors and introduce a new learning paradigm
                            -- {' '}<strong>VisualSplit</strong>, to explore the representation learning capacity using
                            such simple descriptors.
                        </li>
                        <li>
                            Aligning closely with human perceptual understanding, the proposed approach highlights the
                            potential of the overlooked classical while effective visual descriptors.
                        </li>
                        <li>
                            Extensive experimental analysis on low-level and high-level vision tasks, covering various
                            applications, validates the effectiveness of our {' '}<strong>VisualSplit</strong> learning
                            approach.
                            It shows that precise, independent, and intuitive manipulation of image attributes (such as
                            geometry, colour, and illumination) can be achieved.
                        </li>
                    </ul>
                </section>

                {/* Method Overview */}
                <section id="method" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">Method at a Glance</h3>
                    <img
                        src={framework_path}
                        alt="Framework overview"
                        className="w-full rounded-xl mb-6"
                    />
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="rounded-2xl border p-4">
                            <h4 className="font-semibold mb-2">Inputs</h4>
                            <p className="text-sm">
                                <strong>Edges:</strong> structural contours (e.g., Sobel/Canny).
                                <br/>
                                <strong>Colour Segmentation:</strong> region-level appearance
                                with a small number of colour clusters (choose K).
                                <br/>
                                <strong>Grayscale Histogram:</strong> global luminance
                                statistics (e.g., 100 bins).
                            </p>
                        </div>
                        <div className="rounded-2xl border p-4">
                            <h4 className="font-semibold mb-2">Backbone</h4>
                            <p className="text-sm">
                                A ViT encoder consumes local tokens (edges/segments) and a
                                global conditioning vector (histogram). Cross-attention and
                                AdaLN-style conditioning inject descriptor information into
                                learnable image tokens.
                            </p>
                        </div>
                        <div className="rounded-2xl border p-4">
                            <h4 className="font-semibold mb-2">Objective</h4>
                            <p className="text-sm">
                                The decoder reconstructs the RGB image. The encoder is thereby
                                trained to bind structure, region colour, and luminance in a
                                controllable yet compact representation—without text prompts or
                                masking schedules.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Experiments: Descriptor-to-Image */}
                <section id="experiments" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">
                        Descriptor-to-Image: Visual Examples
                    </h3>
                    <p className="mb-4 text-sm">
                        The examples below illustrate how each descriptor contributes to the
                        reconstruction. Replace the images with your own results placed in{" "}
                        <code>public/</code>.
                    </p>

                    <div className="flex flex-col md:flex-row items-center md:items-start gap-4">
                        <figure className="rounded-xl border p-2 md:w-1/3">
                            <img
                                src={experimentImages.original}
                                alt="Original image"
                                className="w-full object-cover rounded-lg"
                            />
                            <figcaption className="mt-2 text-center text-xs">Original</figcaption>
                        </figure>
                        <figure className="rounded-xl border p-2 md:w-1/3 flex flex-col items-center">
                            <div className="grid grid-cols-2 gap-2 w-full">
                                <img
                                    src={experimentImages.edges}
                                    alt="Edge map"
                                    className="w-full object-cover rounded-lg"
                                />
                                <img
                                    src={experimentImages.segmentation}
                                    alt="Colour segmentation"
                                    className="w-full object-cover rounded-lg"
                                />

                                <img
                                    src={experimentImages.histogram}
                                    alt="Grayscale histogram visualisation"
                                    className="pt-10 w-full object-cover rounded-lg col-span-2"
                                />

                            </div>
                            <figcaption className="mt-2 text-center text-xs">
                                Visual Descriptors
                            </figcaption>
                        </figure>
                        <figure className="rounded-xl border p-2 md:w-1/3">
                            <img
                                src={experimentImages.artist}
                                alt="Human illustration"
                                className="w-full object-cover rounded-lg"
                            />
                            <figcaption className="mt-2 text-center text-xs">Artist Illustration</figcaption>
                        </figure>
                        <figure className="rounded-xl border p-2 md:w-1/3">
                            <img
                                src={experimentImages.output}
                                alt="VisualSplit"
                                className="w-full object-cover rounded-lg"
                            />
                            <figcaption className="mt-2 text-center text-xs">VisualSplit</figcaption>
                        </figure>
                    </div>
                    <p className="mt-2 text-xs text-center">
                        We thank Jing Dong for the illustration; used with permission.
                    </p>
                </section>

                {/* Independent Controls */}
                <section id="controls" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">Independent Controls</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <ImageSlider
                            title="Illumination (Histogram)"
                            description={`Modify only the grayscale histogram while keeping edges and colour segmentation fixed. The reconstruction changes global brightness/contrast without altering geometry or region colours.`}
                            beforeSrc={editImages.illuminationBefore}
                            options={[
                                {label: 'Low-light', src: editImages.illuminationLow},
                                {label: 'Normal', src: editImages.illuminationNormal},
                                {label: 'High exposure', src: editImages.illuminationHigh},
                            ]}
                        />
                        <ImageSlider
                            title="Colour (Segmentation)"
                            description={`Re-assign colours to selected segments while keeping edges and histogram fixed. The reconstruction updates region appearance consistently, leaving structure and luminance intact.`}
                            beforeSrc={editImages.colourBefore}
                            options={[
                                {label: 'f=0.25', src: editImages.colour025},
                                {label: 'f=0.75', src: editImages.colour075},
                                {label: 'f=1', src: editImages.colour100},
                                {label: 'f=1.25', src: editImages.colour125},
                                {label: 'f=1.75', src: editImages.colour175},
                            ]}
                        />
                    </div>
                </section>

                <section id="applications" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">Applications</h3>

                    {/* A. Visual Restoration with Diffusion Models */}
                    <div className="rounded-2xl border p-4 mb-8">
                        <h4 className="font-semibold mb-2">A. Visual Restoration with Diffusion Models</h4>
                        <p className="text-sm mb-3">
                            VisualSplit provides structured global (<em>histogram</em>) and local (<em>edges</em>, <em>colour
                            segments</em>)
                            conditions that plug into Stable Diffusion 1.5. Following the paper, the global
                            representation can be fed
                            as an IP-Adapter–style signal while local tokens drive a ControlNet branch, yielding
                            faithful reconstructions
                            without relying on prompts.
                        </p>
                        {/* Descriptor inputs */}
                        <figure className="mx-auto rounded-xl border p-2 md:w-1/3 flex flex-col items-center">
                            <img
                                src={appAssets.restorationCond}
                                alt="Edge map"
                                className="w-full object-cover rounded-lg"
                            />
                            <figcaption className="mt-2 text-center text-xs">
                                Visual Descriptors
                            </figcaption>
                        </figure>
                        <OtherModelSlider options={otherModelOptions} ours={appAssets.restorationOurs}
                                          gt={appAssets.restorationGT}/>

                        {/* Metrics table (from Table 3) */}
                        <RestorationTable />
                    </div>

                    {/* B. Descriptor-Guided Editing (No retraining) */}
                    <div className="rounded-2xl border p-4 mb-8">
                        <h4 className="font-semibold mb-2">B. Descriptor-Guided Editing (No Retraining)</h4>
                        <p className="text-sm mb-3">
                            Editing is performed by modifying individual descriptors while keeping others fixed. Below
                            we show two independent edits: altering the <em>colour segmentation map</em> for region
                            recolouring and changing the <em>gray-level histogram</em> for global luminance. Other
                            descriptors remain unchanged, preserving geometry and non-target attributes.
                        </p>


                        <h5 className="font-medium mb-2">1. Gray-Level Histogram Editing</h5>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingEdgeHist} alt="Edge map"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Edge</figcaption>
                            </figure>
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingSegHist} alt="Colour segmentation map"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Colour Segmentation</figcaption>
                            </figure>
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingOrig} alt="Ground truth"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Ground Truth</figcaption>
                            </figure>
                        </div>
                        <HistogramSlider before={appAssets.editingHistBefore} options={histogramOptions} />

                        <h5 className="font-medium mb-2">2. Colour Map Editing</h5>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingEdgeColour} alt="Edge map"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Edge</figcaption>
                            </figure>
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingSegColourOrig} alt="Colour map (Original)"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Colour Map (Original)</figcaption>
                            </figure>
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingSegColourEdited} alt="Colour map (Edited)"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Colour Map (Edited)</figcaption>
                            </figure>
                            <figure className="rounded-xl border p-2">
                                <img src={appAssets.editingHistColour} alt="Gray-level histogram"
                                     className="w-full object-cover rounded-lg"/>
                                <figcaption className="mt-2 text-center text-xs">Gray-Level Histogram</figcaption>
                            </figure>
                        </div>
                        <OtherModelSlider options={editingModelOptions} ours={appAssets.editingColourOut}
                                          gt={appAssets.editingColourOrig}/>
                        <div className="text-right mb-4">
                            <Link href="/colour-map-examples">
                                <button className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md">
                                    View more examples
                                </button>
                            </Link>
                        </div>

                    </div>


                </section>


                {/*/!* Poster *!/*/}
                {/*<section id="poster" className="mb-12">*/}
                {/*    <h3 className="text-2xl font-semibold mb-4">Poster</h3>*/}
                {/*    <div className="w-full h-[800px]">*/}
                {/*        <iframe*/}
                {/*            src={links.poster}*/}
                {/*            title="VisualSplit poster"*/}
                {/*            className="w-full h-full border rounded"*/}
                {/*        />*/}
                {/*    </div>*/}
                {/*</section>*/}

                {/* BibTeX */}
                <section id="bibtex" className="mb-12">
                    <h3 className="text-2xl font-semibold mb-4">BibTeX</h3>
                    <pre
                        id="bibtex"
                        className="bg-gray-50 p-3 rounded-lg overflow-x-auto text-xs"
                    >
{`@inproceedings{Qu2025VisualSplit,
  title   = {Exploring Image Representation with Decoupled Classical Visual Descriptors},
  author  = {Qu, Chenyuan and Chen, Hao and Jiao, Jianbo},
  booktitle = {British Machine Vision Conference (BMVC)},
  year    = {2025}
}`}
          </pre>
                </section>

                {/* Footer */}
                <footer className="border-t pt-6 text-center text-sm text-gray-500">
                    <p>
                        &copy; 2025 VisualSplit. This page is an unofficial summary for the
                        BMVC 2025 paper. Replace placeholders with your final assets and
                        links before release.
                    </p>
                    <p className="mt-2">
                        Contact:{" "}
                        {authors
                            .map((a) => (a.email ? `${a.name} <${a.email}>` : a.name))
                            .join(" · ")}
                    </p>
                </footer>
            </main>
        </>
    );
}
