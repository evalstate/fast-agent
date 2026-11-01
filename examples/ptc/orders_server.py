"""Order analytics MCP server for the programmatic tool calling (PTC) demo.

The tools are deliberately fine-grained: answering a question such as "which EMEA customer
has the most delivered revenue?" takes one call per customer. With direct tool calling every
result flows through the model's context; with `--ptc` the model writes one Python script that
fans the calls out concurrently and returns only the answer.

Run with:
    uv run orders_server.py
"""

from __future__ import annotations

import random
from typing import Literal

from fastmcp import FastMCP
from pydantic import BaseModel

Region = Literal["EMEA", "AMER", "APAC"]
Status = Literal["delivered", "shipped", "cancelled", "returned"]

app = FastMCP(name="orders")


class Customer(BaseModel):
    id: str
    name: str
    region: Region


class Order(BaseModel):
    id: str
    customer_id: str
    total: float
    status: Status
    month: str


class CustomerList(BaseModel):
    customers: list[Customer]


class OrderList(BaseModel):
    customer_id: str
    orders: list[Order]


def _dataset() -> tuple[list[Customer], dict[str, list[Order]]]:
    rng = random.Random(1234)
    names = [
        "Acme",
        "Globex",
        "Initech",
        "Umbrella",
        "Hooli",
        "Stark",
        "Wayne",
        "Wonka",
        "Tyrell",
        "Cyberdyne",
        "Soylent",
        "Aperture",
        "Vandelay",
        "Gringotts",
        "Oscorp",
    ]
    regions: list[Region] = ["EMEA", "AMER", "APAC"]
    statuses: list[Status] = [
        "delivered",
        "delivered",
        "delivered",
        "shipped",
        "cancelled",
        "returned",
    ]
    customers = [
        Customer(id=f"C{index:03d}", name=name, region=regions[index % 3])
        for index, name in enumerate(names, start=1)
    ]
    orders: dict[str, list[Order]] = {}
    for customer in customers:
        orders[customer.id] = [
            Order(
                id=f"{customer.id}-O{number:02d}",
                customer_id=customer.id,
                total=round(rng.uniform(40, 2500), 2),
                status=rng.choice(statuses),
                month=f"2026-{rng.randint(1, 9):02d}",
            )
            for number in range(1, rng.randint(6, 14))
        ]
    return customers, orders


CUSTOMERS, ORDERS = _dataset()


@app.tool(description="List customers, optionally filtered by sales region.")
def list_customers(region: Region | None = None) -> CustomerList:
    return CustomerList(
        customers=[c for c in CUSTOMERS if region is None or c.region == region],
    )


@app.tool(description="Get every order (id, total in USD, status, month) for one customer.")
def get_orders(customer_id: str) -> OrderList:
    if customer_id not in ORDERS:
        raise ValueError(f"Unknown customer_id {customer_id!r}")
    return OrderList(customer_id=customer_id, orders=ORDERS[customer_id])


if __name__ == "__main__":
    app.run()
