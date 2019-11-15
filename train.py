# import witwidget
import os
import ke
import sk
# from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

from pathlib import Path
import config

dir = Path().absolute()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(dir, config.KEYFILE)


if __name__ == '__main__':
    sk.train_sklearn()
    ke.train_keras()

