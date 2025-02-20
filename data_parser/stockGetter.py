import yfinance as yf
from datetime import datetime
from pandas.core.series import Series


class Stock:
    """
    Serves as a way to get stock data from Yahoo's API.
    """
    def __init__(
            self,
            name: str,
            start_date: str,
            end_date: str,
            interval: str = "1d"):
        """
        A way to initializes a Stock object with a name,
        start date, end date, and optional interval.
        
        :param name: The name of the stock or financial
        instrument for which you want to download data
        :type name: str
        :param start_date: The start date for downloading stock data. 
        :type start_date: str
        :format start_date: "yyyy-dd-mm"
        :param end_date: The end date for the data you want to download.
        :type end_date: str
        :format end_date: "yyyy-dd-mm"
        :param interval: The interval you want
        :type interval: str
        """
        self.name = name
        self.start_date = start_date 
        self.end_date = end_date

        # Avoiding chosing interval of 1h
        if interval == "1h":
            self.interval = "60m"
        else:
            self.interval = interval

    def get_data(
            self
            ) -> tuple[
                datetime,
                Series,
                Series,
                Series,
                Series
                ]:
        """
        The method returns the stock prices.

        :return: The stock data
        :type return: tuple[
                datetime,
                Series,
                Series,
                Series,
                Series
                ]
        """
        # Getting the dates of the stock data
        stock_data = yf.download(
            self.name,
            self.start_date,
            self.end_date,
            interval=self.interval
            )
        
         # Getting the dates of the stock data
        dates = stock_data.index

        return (
            dates,
            stock_data['Open'],
            stock_data['High'],
            stock_data['Low'],
            stock_data['Close']
            )
    