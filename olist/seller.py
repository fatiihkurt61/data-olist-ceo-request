
import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy(
        )  # Make a copy before using inplace=True so as to avoid modifying self.data
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(
            inplace=True)  # There can be multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """Satıcı başına kargoya teslim gecikmesi ve bekleme süresi."""
        order_items = self.data["order_items"].copy()

        orders = self.data["orders"].query(
            "order_status == 'delivered'"
        ).copy()

        ship = order_items.merge(
            orders,
            on="order_id",
            how="inner"
        )

        date_columns = [
            "shipping_limit_date",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_purchase_timestamp"
        ]

        for column in date_columns:
            ship[column] = pd.to_datetime(
                ship[column], errors="coerce"
            )

        ship["carrier_difference"] = (
            ship["order_delivered_carrier_date"]
            - ship["shipping_limit_date"]
        ).dt.total_seconds() / 86400

        ship["wait_time"] = (
            ship["order_delivered_customer_date"]
            - ship["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

        result = ship.groupby(
            "seller_id", as_index=False
        ).agg(
            delay_to_carrier=("carrier_difference", "mean"),
            wait_time=("wait_time", "mean")
        )

        result["delay_to_carrier"] = (
            result["delay_to_carrier"].clip(lower=0)
        )

        return result

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date) /
                np.timedelta64(24, 'h'))
            if days > 0:
                return days
            else:
                return 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp)
                / np.timedelta64(24, 'h'))
            return days

        delay = ship.groupby('seller_id')\
                    .apply(delay_to_logistic_partner)\
                    .reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id')\
                   .apply(order_wait_time)\
                   .reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')

        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        # First, get only orders that are approved
        orders_approved = self.data['orders'][[
            'order_id', 'order_approved_at'
        ]].dropna()

        # Then, create a (orders <> sellers) join table because a seller can appear multiple times in the same order
        orders_sellers = orders_approved.merge(self.data['order_items'],
                                               on='order_id')[[
                                                   'order_id', 'seller_id',
                                                   'order_approved_at'
                                               ]].drop_duplicates()
        orders_sellers["order_approved_at"] = pd.to_datetime(
            orders_sellers["order_approved_at"])

        # Compute dates
        orders_sellers["date_first_sale"] = orders_sellers["order_approved_at"]
        orders_sellers["date_last_sale"] = orders_sellers["order_approved_at"]
        df = orders_sellers.groupby('seller_id').agg({
            "date_first_sale": "min",
            "date_last_sale": "max"
        })
        df['months_on_olist'] = round(
            (df['date_last_sale'] - df['date_first_sale']) /
            np.timedelta64(30, 'D'))
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data['order_items']

        n_orders = order_items.groupby('seller_id')['order_id']\
            .nunique()\
            .reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = order_items.groupby('seller_id', as_index=False).agg(
            {'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return self.data['order_items'][['seller_id', 'price']]\
            .groupby('seller_id')\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_review_score(self):
        """Satıcı başına yıldız oranlarını ve ortalama puanı döndür."""
        order_sellers = self.data["order_items"][
            ["order_id", "seller_id"]
        ].drop_duplicates()

        reviews = self.order.get_review_score()

        seller_reviews = order_sellers.merge(
            reviews,
            on="order_id",
            how="inner"
        )

        return seller_reviews.groupby(
            "seller_id", as_index=False
        ).agg(
            share_of_five_stars=("dim_is_five_star", "mean"),
            share_of_one_stars=("dim_is_one_star", "mean"),
            review_score=("review_score", "mean")
        )

    def get_training_data(self):
        """Seller bazında gelir, yorum maliyeti ve kâr tablosu döndür."""

        # Temel seller bilgileri ve satış metrikleri
        training_set = (
            self.get_seller_features()
            .merge(self.get_seller_delay_wait_time(), on="seller_id")
            .merge(self.get_active_dates(), on="seller_id")
            .merge(self.get_quantity(), on="seller_id")
            .merge(self.get_sales(), on="seller_id")
        )

        # Yalnızca teslim edilmiş siparişlerde ürün fiyatının %10'u gelir.
        delivered_orders = self.data["orders"].loc[
            self.data["orders"]["order_status"].eq("delivered"),
            ["order_id"],
        ]
        delivered_items = self.data["order_items"].merge(
            delivered_orders,
            on="order_id",
            how="inner",
        )
        sales_fees = (
            delivered_items.assign(sales_fee=delivered_items["price"] * 0.10)
            .groupby("seller_id", as_index=False)["sales_fee"]
            .sum()
        )

        # Yorum maliyetini, bir siparişteki satıcılar arasında paylaştır.
        order_sellers = self.data["order_items"][
            ["order_id", "seller_id"]
        ].drop_duplicates()
        order_sellers["seller_count"] = (
            order_sellers.groupby("order_id")["seller_id"].transform("count")
        )

        reviews = self.order.get_review_score()[
            ["order_id", "review_cost"]
        ]
        seller_review_costs = order_sellers.merge(
            reviews,
            on="order_id",
            how="inner",
        )
        seller_review_costs["allocated_review_cost"] = (
            seller_review_costs["review_cost"]
            / seller_review_costs["seller_count"]
        )
        seller_review_costs = (
            seller_review_costs.groupby("seller_id", as_index=False)
            ["allocated_review_cost"]
            .sum()
            .rename(columns={"allocated_review_cost": "cost_of_reviews"})
        )

        # Seller başına aylık 80 BRL abonelik geliri.
        training_set["months_on_olist"] = (
            training_set["months_on_olist"].fillna(0).clip(lower=1)
        )
        training_set["subscription_revenue"] = (
            training_set["months_on_olist"] * 80
        )

        training_set = (
            training_set
            .merge(sales_fees, on="seller_id", how="left")
            .merge(seller_review_costs, on="seller_id", how="left")
        )

        training_set[["sales_fee", "cost_of_reviews"]] = (
            training_set[["sales_fee", "cost_of_reviews"]].fillna(0)
        )
        training_set["revenues"] = (
            training_set["subscription_revenue"] + training_set["sales_fee"]
        )
        training_set["profits"] = (
            training_set["revenues"] - training_set["cost_of_reviews"]
        )

        return training_set