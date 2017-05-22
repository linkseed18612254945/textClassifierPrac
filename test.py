
import logging
import time
date = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logging.basicConfig(level=logging.INFO,
                    filename='./log/%s.txt' % date,
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


logging.warning('w')
