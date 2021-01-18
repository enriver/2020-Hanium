import schedule
import time
import os

os.system("sudo timedatectl set-timezone 'Asia/Seoul' ")

def job():
    print('model starting')
    try:
    
        os.system('python train.py')

    except:
        print('error')
        os.system('python train.py')


schedule.every().monday.at('18:00').do(job)
schedule.every().tuesday.at('18:00').do(job)
schedule.every().wednesday.at('18:00').do(job)
schedule.every().thursday.at('18:00').do(job)
schedule.every().sunday.at('18:00').do(job)



while True:
    schedule.run_pending()
    time.sleep(1)

