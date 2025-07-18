import yfinance as yf
from datetime import datetime
from pandas.core.series import Series
import pandas as pd


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
        self._validate_date(start_date)
        self._validate_date(end_date)
        
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

        :return: The stock data in the following format
            tuple[
                Date,
                Open,
                High,
                Low,
                Close,
                ]
        :type return: tuple[
                datetime,
                Series,
                Series,
                Series,
                Series,
                ]
        """
        # Getting the dates of the stock data
        stock_data = yf.download(
            self.name,
            self.start_date,
            self.end_date,
            interval=self.interval
            )
        
        # Check if stock_data is empty
        if stock_data.empty:
            raise ValueError(f"No data found for stock {self.name} between {self.start_date} and {self.end_date}")
        
        # Getting the dates of the stock data
        dates = stock_data.index

        open_ = pd.Series(stock_data['Open'].values.T[0].tolist())
        high = pd.Series(stock_data['High'].values.T[0].tolist())
        low = pd.Series(stock_data['Low'].values.T[0].tolist())
        close = pd.Series(stock_data['Close'].values.T[0].tolist())
        return (
            dates,
            open_,
            high,
            low,
            close
        )
    
    def get_all_data(
            self
            ) -> tuple[
                datetime,
                Series,
                Series,
                Series,
                Series,
                Series
                ]:
        """
        The method returns the stock prices.

        :return: The stock data in the following format
            tuple[
                Date,
                Open,
                High,
                Low,
                Close,
                Volume
                ]
        :type return: tuple[
                datetime,
                Series,
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

        open_ = pd.Series(stock_data['Open'].values.T[0].tolist())
        high = pd.Series(stock_data['High'].values.T[0].tolist())
        low = pd.Series(stock_data['Low'].values.T[0].tolist())
        close = pd.Series(stock_data['Close'].values.T[0].tolist())
        volume = pd.Series(stock_data['Volume'].values.T[0].tolist())

        return (
            dates,
            open_,
            high,
            low,
            close,
            volume
            )
    
    def _validate_date(self, date: str) -> None:
        """
        Validates the date.

        :param date: the date
        :type date: str
        :raises TypeError: wrong date format
        """
        if not isinstance(date, str):
            raise TypeError(
                "You must provide type=`str` as date in the form:"
                f" yyyy-mm-dd. You provided: type=`{type(date)}`")
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except TypeError("The date must be of the form `yyyy-mm-dd`!") as e:
            raise e
    