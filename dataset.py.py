### Create file named dataset.py
### Paste this inside dataset.py



# coding=utf-8
import json
import os
from pathlib import Path
import datasets
from PIL import Image
import pandas as pd

logger = datasets.logging.get_logger(__name__)
_CITATION = """{}"""
_DESCRIPTION = """Discharge Summary"""


def load_image(image_path):
    image = Image.open(image_path)
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


class SroieConfig(datasets.BuilderConfig):
    """BuilderConfig for SROIE"""
    def __init__(self, **kwargs):
        """BuilderConfig for SROIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SroieConfig, self).__init__(**kwargs)


class Sroie(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SroieConfig(name="discharge", version=datasets.Version("1.0.0"), description="Discharge summary dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['others',
                                    'produttore_key',
                                    'produttore_value',
                                    'cliente_key',
                                    'cliente_value',
                                    'unitloc_key',
                                    'unitloc_value',
                                    'operatore_key',
                                    'operatore_value',
                                    'referente_key',
                                    'referente_value',
                                    'cfproduttore_key',
                                    'cfproduttore_value',
                                    'telefono_key',
                                    'telefono_value',
                                    'emailcliente_key',
                                    'emailcliente_value',
                                    'datarichiesta_key',
                                    'datarichiesta_value',
                                    'orariorichiesta_key',
                                    'orariorichiesta_value',
                                    'emailproduttore_key',
                                    'emailproduttore_value',
                                    'mattina_key',
                                    'mattina_value',
                                    'pomeriggio_key',
                                    'pomeriggio_value',
                                    'cer_key',
                                    'cer_value',
                                    'descrizione_key',
                                    'descrizione_value',
                                    'sf_key',
                                    'sf_value',
                                    'classpericolo_key',
                                    'destino_key',
                                    'destino_value',
                                    'confezionamento_key',
                                    'confezionamento_value',
                                    'destinazione_key',
                                    'destinazione_value'
                                    ]
                            )
                    ),
                    #"image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        #downloaded_file = dl_manager.download_and_extract(_URLS)
        # move files from the second URL together with files from the first one.
        dest = Path('dataset')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest/"train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest/"test"}
            ),
        ]

    def _generate_examples(self, filepath):

        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotation_dir")
        img_dir = os.path.join(filepath, "img_dir")

        for guid, fname in enumerate(sorted(os.listdir(img_dir))):

            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".csv")


            df = pd.read_csv(file_path)

            image_path = os.path.join(img_dir, fname)

            image, size = load_image(image_path)

            boxes = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in zip(df['left'],df['top'],df['left']+df['width'],df['top']+df['height'])]
            text = [i for i in df['text']]
            label = [i for i in df['label']]

            boxes = [normalize_bbox(box, size) for box in boxes]

            print(image_path)
            for i in boxes:
              for j in i:
                if j>1000:
                  print(j)
                  pass

            yield guid, {"id": str(guid), "words": text, "bboxes": boxes, "ner_tags": label, "image_path": image_path}
