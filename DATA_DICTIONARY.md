Необходимо построить алгоритм прогнозирования спроса: определить, сколько и какого товара необходимо привезти в каждый магазин, в каждый день на 1 неделю вперёд, используя предоставленные данные. 
Тестовый период: данные сентября 2024 года. Прогноз нужно строить только на те даты, которые есть в тестовой выборке, чтобы была возможность посчитать метрики и оценить качество.
Прогнозировать нужно суммарные онлайн+оффлайн продажи.


При решении необходимо:
- провести EDA, построить графики
- построить модель прогнозирования 
- подобрать метрику для оценки результатов.

Подробные комментарии с ходом мыслей и идеями на будущее (что ещё можно проверить в данных, какие фичи ещё стоит добавить, какие модели затестить и тп) будут жирным плюсом. 


Оформить решение необходимо в виде ноутбука с комментариями (либо скрипты + ноутбук с вызовом методов и с комментариями)





About Dataset
Dataset Description

This dataset contains sales information from four stores of one of the retailers over 25 months. Participants are expected to use these files to develop models that can predict customer demand. Additionally, the dataset includes a holdout sample with sales data for a 1-month period for which forecasts should be provided.

sales.csv

    Purpose: This file contains aggregated store sales for specific dates.
    Columns:
        date: Sales date
        item_id: A unique identifier for each product
        quantity: Total quantity of product sold per day
        price_base: Average sales price per day
        sum_total: Total daily sales amount
        store_id: Store number

online.csv

    Purpose: This file contains aggregated online sales by store for specific dates.
    Columns:
        date: Sales date
        item_id: A unique identifier for each product
        quantity: Total quantity of product sold per day (online)
        price_base: Average sales price per day
        sum_total: Total daily sales amount
        store_id: Store number

markdowns.csv

    Purpose: This file provides data on products sold at markdown prices in each store.
    Columns:
        date: Date of markdown
        item_id: A unique identifier for each product
        normal_price: Regular price
        price: Price during markdown
        quantity: Quantity sold at markdown
        store_id: Store number

price_history.csv

    Purpose: This file contains price changes data in each store.
    Columns:
        date: Date of price change
        item_id: A unique identifier for each product
        price: Item new price
        code: Price change code
        store_id: Store number

discounts_history.csv

    Purpose: Contains historical promo data for each specific store.
    Columns:
        date: Date
        item_id: A unique identifier for each product
        sale_price_before_promo: Price before promo period started
        sale_price_time_promo: Price during the promo period
        promo_type_code: Promo code type
        doc_id: Promo document number
        number_disc_day: Sequential day number of the current promo period
        store_id: Store number

actual_matrix.csv

    Purpose: Contains the list of products available in stores.
    Columns:
        item_id: A unique identifier for each product
        date: Date of last product appearance in the current matrix
        store_id: Store number

catalog.csv

    Purpose: Product catalog with characteristics.
    Columns:
        item_id: A unique identifier for each product
        dept_name: Product department (hierarchy level)
        class_name: Product class (hierarchy level)
        subclass_name: Product subclass (hierarchy level)
        item_type: Product type
        weight_volume: Volumetric weight
        weight_netto: Net weight
        fatness: Fat content

stores.csv

    Purpose: Contains stores info data.
    Columns:
        store_id: Store number
        division: Store division
        format: Store format
        city: Location
        area: Store sales area

