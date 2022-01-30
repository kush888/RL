from binance import Client
import pandas as pd
import datetime

api_key = "IkglEvMVJST0OmJA3Jfhi7nGUirfrYRnGsGdBTUoKNkpOPiDmSnfElk3zujUrabT"
secret_key = "hXOnb96VFSBSfvrHJYAdBv9UGR61CnbpqXZpDhoqGqc0QxbLNI9BdsCZsRrtyou2"
client = Client(api_key, secret_key)


def get_data(symbol, interval, sinceThisDate, untilThisDate):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, str(sinceThisDate), str(untilThisDate)))
    return frame


untilThisDate = datetime.datetime.now()
sinceThisDate = datetime.datetime.now() - datetime.timedelta(days=1)

fileName = "test"
frame = get_data("BTCUSDT", "1m", sinceThisDate, untilThisDate)
frame.to_pickle(fileName)

print (pd.read_pickle("test"))
