import os
import json
import argparse
import itertools
from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_coco(dt_path, coco_api):
    assert os.path.isfile(dt_path), dt_path + " does not exist!"
    with open(dt_path, "r") as f:
        lines = f.readlines()
    predictions = [json.loads(line.strip('\n')) for line in lines]
    coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
    coco_gt = coco_api
    coco_dt = coco_api.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
    line = "Evaluation results for bbox: \n" + create_small_table(results)
    print(line)    
    return line


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a json result file with iou match')
    parser.add_argument('--detfile', required=True, help='path of json result file to load')
    parser.add_argument('--gtfile', required=True, help='path of json coco gt file to load')
    args = parser.parse_args()
    coco_api = COCO(args.gtfile)
    eval_coco(args.detfile, coco_api)
