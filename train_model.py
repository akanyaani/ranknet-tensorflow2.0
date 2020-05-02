import os

import click

from data_pipeline import *
from ranknet import LTRModelRanknet


@click.command()
@click.option('--data-path', type=str, default="./data/tf_records",
			  show_default=True,
			  help="out directory")
@click.option('--out-dir', type=str, default="/media/akanyaani/Disk2/ranknet",
			  show_default=True,
			  help="tf records path")
@click.option('--exp-name', type=str, default="ltr", show_default=True, help="exp name")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--window-size', type=int, default=512, show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=8, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=5e-4, show_default=True, help="learning rate")
@click.option('--graph-mode', type=bool, default=True, show_default=True, help="graph execution")
def train(data_path, out_dir, exp_name, optimizer, window_size, batch_size, learning_rate, graph_mode):
	MODEL_DIR = out_dir + "/models/" + exp_name
	LOG_DIR = MODEL_DIR + "/log/"

	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)

	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)

	train_tf_records = data_path + "/train"
	test_tf_records = data_path + "/test"

	train_dataset = pairwise_batch_iterator(train_tf_records, window_size, batch_size, no_threads=8)
	test_dataset = pairwise_batch_iterator(test_tf_records, window_size, batch_size, no_threads=2)

	model = LTRModelRanknet(learning_rate=learning_rate)
	model.ranknet_type = "factor"
	model.create_optimizer(optimizer_type=optimizer)
	model.create_checkpoint_manager(MODEL_DIR)
	model.create_summary_writer(LOG_DIR)
	model.log_dir = LOG_DIR

	model.fit([train_dataset, test_dataset], graph_mode)
	print("Training Done............")


if __name__ == "__main__":
	train()
