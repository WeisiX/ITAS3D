import os
from download_gdrive import *

#download single-level translation checkpoints
file_id = '1ulZalAiXj-_F60aI7iK5KFBO0JUV-9Pw'
chpt_path = '../img_translation/checkpoints/'
destination = os.path.join(chpt_path, 'frame_seq_630_t1.zip')
download_file_from_google_drive(file_id, destination) 