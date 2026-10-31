"""Build order-level features for Olist."""

import pandas as pd

from olist.data import Olist


class Order:
    """Provide order-level feature tables."""

    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """Return delivery durations in fractional days."""
        orders = self.data["orders"].copy()

        if is_delivered:
            orders = orders.loc[
                orders["order_status"] == "delivered"
            ].copy()

        date_columns = [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        for column in date_columns:
            orders[column] = pd.to_datetime(orders[column])

        orders["wait_time"] = (
            orders["order_delivered_customer_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

        orders["expected_wait_time"] = (
            orders["order_estimated_delivery_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

        orders["delay_vs_expected"] = (
            (
                orders["order_delivered_customer_date"]
                - orders["order_estimated_delivery_date"]
            ).dt.total_seconds() / 86400
        ).clip(lower=0)

        return orders[
            [
                "order_id",
                "wait_time",
                "expected_wait_time",
                "delay_vs_expected",
                "order_status",
            ]
        ]

    def get_review_score(self):
        """Yorum puanlarını ve puana bağlı tahmini maliyeti döndür."""
        reviews = self.data["order_reviews"].copy()

        reviews["dim_is_five_star"] = (
            reviews["review_score"].eq(5).astype(int)
        )
        reviews["dim_is_one_star"] = (
            reviews["review_score"].eq(1).astype(int)
        )
        reviews["review_cost"] = (
            reviews["review_score"]
            .map({1: 100, 2: 50, 3: 40})
            .fillna(0)
        )

        return reviews[
            [
                "order_id",
                "dim_is_five_star",
                "dim_is_one_star",
                "review_score",
                "review_cost",
            ]
        ]

    def get_number_items(self):
        """Return the total item count per order."""
        return (
            self.data["order_items"]
            .groupby("order_id")
            .size()
            .reset_index(name="number_of_items")
        )

    def get_number_sellers(self):
        """Return the unique seller count per order."""
        return (
            self.data["order_items"]
            .groupby("order_id")["seller_id"]
            .nunique()
            .reset_index(name="number_of_sellers")
        )

    def get_price_and_freight(self):
        """Return total price and freight per order."""
        return (
            self.data["order_items"]
            .groupby("order_id")[["price", "freight_value"]]
            .sum()
            .reset_index()
        )

    def get_distance_seller_customer(self):
        """Return average seller-customer distance per order."""
        raise NotImplementedError(
            "The optional distance feature is not implemented yet."
        )

    def get_training_data(
        self,
        is_delivered=True,
        with_distance_seller_customer=False,
    ):
        """Return a combined order feature table without missing values."""
        training_data = (
            self.get_wait_time(is_delivered=is_delivered)
            .merge(
                self.get_review_score(),
                on="order_id",
                how="inner",
            )
            .merge(
                self.get_number_items(),
                on="order_id",
                how="inner",
            )
            .merge(
                self.get_number_sellers(),
                on="order_id",
                how="inner",
            )
            .merge(
                self.get_price_and_freight(),
                on="order_id",
                how="inner",
            )
        )

        if with_distance_seller_customer:
            training_data = training_data.merge(
                self.get_distance_seller_customer(),
                on="order_id",
                how="inner",
            )

        return (
            training_data
            .dropna()
            .reset_index(drop=True)
        )