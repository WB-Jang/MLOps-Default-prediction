"""Sample data generator for testing the MLOps pipeline."""
import json
import random
from datetime import datetime, timedelta

import numpy as np
import psycopg2
from psycopg2.extras import Json


def generate_sample_data(n_samples=1000, default_rate=0.15):
    """
    Generate synthetic loan default data for testing.

    Args:
        n_samples: Number of samples to generate
        default_rate: Rate of defaults (0-1)

    Returns:
        List of tuples (data_date, cat_features, num_features, target)
    """
    samples = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(n_samples):
        # Generate random date within last year
        days_offset = random.randint(0, 365)
        data_date = base_date + timedelta(days=days_offset)

        # Generate categorical features
        cat_features = {
            "loan_type": random.randint(0, 4),  # 5 types of loans
            "employment_status": random.randint(0, 3),  # 4 employment statuses
            "education_level": random.randint(0, 5),  # 6 education levels
            "marital_status": random.randint(0, 2),  # 3 marital statuses
            "property_ownership": random.randint(0, 2),  # 3 property types
            "credit_history": random.randint(0, 4),  # 5 credit history levels
            "loan_purpose": random.randint(0, 7),  # 8 loan purposes
            "region": random.randint(0, 9),  # 10 regions
        }

        # Generate numerical features
        # Features that influence default
        income = random.uniform(20000, 200000)
        loan_amount = random.uniform(5000, 100000)
        debt_to_income = loan_amount / (income + 1)
        credit_score = random.randint(300, 850)

        # Determine if default based on risk factors
        risk_score = (
            (debt_to_income * 0.3)
            + ((850 - credit_score) / 550 * 0.3)
            + (loan_amount / 100000 * 0.2)
            + (random.random() * 0.2)
        )

        target = 1 if risk_score > (1 - default_rate) else 0

        num_features = {
            "annual_income": float(income),
            "loan_amount": float(loan_amount),
            "credit_score": float(credit_score),
            "debt_to_income_ratio": float(debt_to_income),
            "employment_length_years": float(random.uniform(0, 30)),
            "number_of_dependents": float(random.randint(0, 5)),
            "monthly_debt_payment": float(random.uniform(100, 5000)),
            "savings_account_balance": float(random.uniform(0, 50000)),
            "age": float(random.randint(18, 75)),
            "loan_term_months": float(random.choice([12, 24, 36, 48, 60, 84, 120])),
        }

        samples.append((data_date.date(), cat_features, num_features, target))

    return samples


def insert_sample_data_to_db(database_url, n_samples=1000):
    """
    Insert sample data into PostgreSQL database.

    Args:
        database_url: PostgreSQL connection URL
        n_samples: Number of samples to generate
    """
    # Parse database URL
    # Format: postgresql://user:pass@host:port/dbname
    url_parts = database_url.replace("postgresql://", "").split("@")
    user_pass = url_parts[0].split(":")
    host_port_db = url_parts[1].split("/")
    host_port = host_port_db[0].split(":")

    user = user_pass[0]
    password = user_pass[1]
    host = host_port[0]
    port = host_port[1] if len(host_port) > 1 else "5432"
    dbname = host_port_db[1]

    # Connect to database
    conn = psycopg2.connect(
        dbname=dbname, user=user, password=password, host=host, port=port
    )
    cur = conn.cursor()

    # Generate and insert data
    print(f"Generating {n_samples} sample records...")
    samples = generate_sample_data(n_samples)

    print("Inserting into database...")
    inserted = 0
    for data_date, cat_features, num_features, target in samples:
        try:
            cur.execute(
                """
                INSERT INTO loan_data.raw_data 
                (data_date, categorical_features, numerical_features, target)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (data_date, Json(cat_features), Json(num_features), target),
            )
            inserted += 1
        except Exception as e:
            print(f"Error inserting record: {e}")

    conn.commit()
    cur.close()
    conn.close()

    print(f"Successfully inserted {inserted} records!")
    return inserted


if __name__ == "__main__":
    import sys

    # Default database URL
    database_url = "postgresql://mlops_user:mlops_password@localhost:5432/loan_default"

    # Get number of samples from command line
    n_samples = 1000
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])

    if len(sys.argv) > 2:
        database_url = sys.argv[2]

    print("=" * 50)
    print("Sample Data Generator")
    print("=" * 50)
    print(f"Database: {database_url}")
    print(f"Samples to generate: {n_samples}")
    print("=" * 50)

    try:
        inserted = insert_sample_data_to_db(database_url, n_samples)
        print(f"\n✅ Success! Inserted {inserted} sample records.")
        print("\nYou can now:")
        print("  1. View data in PostgreSQL")
        print("  2. Trigger the model_training DAG in Airflow")
        print("  3. Test the API endpoints")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database is initialized (docker-compose up)")
        print("  3. Database URL is correct")
        sys.exit(1)
