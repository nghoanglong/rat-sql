#!/usr/bin/env python

import argparse
import json
import torch

import _jsonnet
import attr
from ratsql.commands import preprocess, train, infer, eval
from ratsql.utils import registry

@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    exp_config = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()

def predict(exp_config, model_config_args, logdir, input_nl, db_id):
    model_config_file = exp_config["model_config"]
    infer_config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))

    inferer = infer.Inferer(infer_config)
    model = inferer.load_model(logdir, exp_config["eval_steps"][0])
    dataset = registry.construct('dataset', inferer.config['data']['test'])

    for _, schema in dataset.schemas.items():
        model.preproc.enc_preproc._preprocess_schema(schema)

    def question(q, db_id):
        spider_schema = dataset.schemas[db_id]
        data_item = SpiderItem(
            text=None, 
            code=None,
            schema=spider_schema,
            orig_schema=spider_schema.orig,
            orig={"question": q}
        )
        model.preproc.clear_items()
        enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
        preproc_data = enc_input, None
        with torch.no_grad():
            return inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
    return question(input_nl, db_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval", choices=["preprocess", "train", "eval", "predict"])
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file,
                                   model_config_args, exp_config, logdir)
        train.main(train_config)
    elif args.mode == "eval":
        for step in exp_config["eval_steps"]:
            infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.infer"
            infer_config = InferConfig(
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                use_heuristic=exp_config["eval_use_heuristic"]
            )
            infer.main(infer_config)

            eval_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.eval"
            eval_config = EvalConfig(
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                infer_output_path,
                eval_output_path
            )
            eval.main(eval_config)

            res_json = json.load(open(eval_output_path))
            print(step, res_json['total_scores']['all']['exact'])
    else:
        # output_file = open(exp_config["logdir"] + "/predicted_file.txt", "w", encoding='utf-8')
        # config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
        # data = json.load(open(config["data"]["test"]["paths"][0]))
        # for value in data:
        #     decoded = predict(exp_config, model_config_args, logdir, value["question"], value["db_id"])
        #     output_file.write(decoded[0]["inferred_code"] + "\n")

        db_id = input('enter database name: ')
        input_nl = input('enter vietnamese question: ')
        decoded = predict(exp_config, model_config_args, logdir, input_nl, db_id)
        logger = train.Logger(exp_config["logdir"] + "/predicted_file.txt", True)
        logger.log(decoded[0]["inferred_code"] + "\n")


if __name__ == "__main__":
    main()
