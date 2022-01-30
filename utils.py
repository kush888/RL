import os
from datetime import datetime

def write_to_file(time, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
#     for i in [net_worth]:
#         time += " {}".format(i)
    print (net_worth)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
#     file.write(time+"\n")
    file.close()