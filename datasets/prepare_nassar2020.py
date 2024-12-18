from detectron2.data import MetadataCatalog, DatasetCatalog
import os

def get_nassar_2020_dicts(img_dir):
  dataset_dicts = []

  for index, image_filename in enumerate(os.listdir(img_dir)):
    dataset_dicts.append({
      "file_name": os.path.join(img_dir, image_filename),
      "image_id": index,
      "height": 256,
      "width": 256,
      "sem_seg_file_name": os.path.join(img_dir, "..", "mask", image_filename)
    })

  return dataset_dicts

def prepare_nassar2020():
  for split_type in ["train", "val"]: #, "test"]:
      DatasetCatalog.register("nassar2020_" + split_type, lambda d=split_type: get_nassar_2020_dicts("datasets/nassar2020/" + split_type))
      MetadataCatalog.get("nassar2020_" + split_type).set(
        stuff_classes=["background", "crop", "weed"], 
        stuff_colors=[(0, 0, 0), (0, 128, 0), (128, 0, 0)],
        ignore_label=255,
        evaluator_type="sem_seg"
        )