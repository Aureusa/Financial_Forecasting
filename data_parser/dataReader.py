from datetime import datetime
from data_parser.stockGetter import Stock
from pandas.core.series import Series

class DataReader:
    """
    Serves as a way to read the stock data from Yahoo's API.
    """
    def __init__(
            self,
            stock_name: str,
            interval: str = "1d"):
        """
        A way of instantiating a DataReader

        :param stock_name: the name of the stock
        :type stock_name: str
        :param end_date: the end date we want our data to be up to,
        defaults to "2024-09-01"
        :type end_date: str, optional
        :param interval: the interval we wish to collect data for,
        defaults to "1d"
        :type interval: str, optional
        """
        self.stock_name = stock_name
        self.interval = interval
        self.data: list[float]|None = None
        self.labels: list[float]|None = None

    def get_data(
            self,
            start_date: str,
            end_date: str
            ) -> tuple[datetime, Series, Series, Series, Series]:
        # Validate dates
        self._validate_date(start_date)
        self._validate_date(end_date)

        # Retrieve data
        dates, open_, high, low, close = self._retrieve_data(start_date, end_date)
        
        # Combine into DOHLC format
        # (dates, open, high, low, close)
        self.data = [
            (dat, op, hi, lo, cl)
            for dat, op, hi, lo, cl in zip(dates, open_, high, low, close)
            ]
        
        if dates is not None:
            msg = f"|| Successfully retrieving data for the period `{start_date}` : `{end_date}` ||"
            border = len(msg) * "="
            message = border + "\n" + msg + "\n" + border
            print(message)
        else:
            msg = f"|| Data retriaval for the period `{start_date}` : `{end_date}` failed! ||"
            border = len(msg) * "="
            message = border + "\n" + msg + "\n" + border
            print(message)
            raise ValueError("DATA FAILIURE!")

        return self.data
    
    def get_all_data(
            self,
            start_date: str,
            end_date: str
            ) -> tuple[datetime, Series, Series, Series, Series, Series]:
        # Validate dates
        self._validate_date(start_date)
        self._validate_date(end_date)

        # Retrieve data
        stock = Stock(
            self.stock_name,
            start_date,
            end_date,
            self.interval
            )
        
        print(f"Retrieving ALL data for the period `{start_date}` : `{end_date}` ...")

        return stock.get_all_data()

    def _retrieve_data(
            self,
            start_date: datetime,
            end_date: datetime
            ) -> tuple[
                datetime,
                Series,
                Series,
                Series,
                Series
                ]:
        """
        Retrieves the date based on a starting date and ending date.

        :param start_date: the starting date
        :type start_date: datetime
        :return: a tuple in the following format:
        Date,
        Open,
        High,
        Low,
        Close.
        :rtype: tuple[
                datetime,
                Series,
                Series,
                Series,
                Series
                ]
        """
        stock = Stock(
            self.stock_name,
            start_date,
            end_date,
            self.interval
            )
        return stock.get_data()

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
        