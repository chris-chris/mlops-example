# import witwidget
import os
import ke
import sk
# from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

from pathlib import Path
import config

from sacred import Experiment
from sacred.observers import MongoObserver
import argparse

dir = Path().absolute()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(dir, config.KEYFILE)

ex = Experiment('wine')
ex.observers.append(MongoObserver.create(url=config.MONGO_URL,
                                         db_name='experiments'))


def get_params():
    parser = argparse.ArgumentParser()
    # gcp ai-platform args
    parser.add_argument("--job-dir", default='.', help="job dir")

    # data hyperparams
    parser.add_argument("--input_size", type=int, default=10)

    parser.add_argument("--framework", type=str, default="sklearn")
    parser.add_argument("--keras_model", type=str, default="dense")
    parser.add_argument("--sklearn_model", type=str, default="linear")
    parser.add_argument("--loss", type=str, default="squared_loss")

    # training hyperparams

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    return args


@ex.config
def hyperparam():
    """hyperparam"""

    args = get_params()

    """@nni.variable(nni.choice('sklearn', 'keras'), name=args.framework)"""
    args.framework = args.framework
    """@nni.variable(nni.choice('linear', 'sgd'), name=args.sklearn_model)"""
    args.sklearn_model = args.sklearn_model
    """@nni.variable(nni.choice('squared_loss', 'huber', 'epsilon_insensitive'), name=args.loss)"""
    args.loss = args.loss
    """@nni.variable(nni.choice(32, 64, 128), name=args.batch_size)"""
    args.batch_size = args.batch_size
    """@nni.variable(nni.loguniform(0.0001, 0.1), name=args.lr)"""
    args.lr = args.lr

    print("hyperparam - ", args)


@ex.automain
def run(args):
    if args.framework == 'sklearn':
        result = sk.train_sklearn(args)
    elif args.framework == 'keras':
        result = ke.train_keras(args)
    else:
        return None

    return result
