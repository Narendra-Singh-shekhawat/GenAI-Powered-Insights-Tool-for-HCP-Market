from postgres_client import PostgresClient


class SalesQueryService:
    def __init__(self):
        self.client = PostgresClient()

    def get_hcps_with_prescription_drop(self, specialty: str, product: str):
        query = """
        WITH sales_trend AS (
            SELECT
                doctor_id,
                doctor_name,
                specialty,
                region,
                product,
                quarter,
                prescriptions,
                LAG(prescriptions) OVER (
                    PARTITION BY doctor_id, product
                    ORDER BY quarter
                ) AS prev_prescriptions
            FROM hcp_sales
            WHERE specialty = %s
              AND product = %s
        )
        SELECT
            doctor_id,
            doctor_name,
            specialty,
            region,
            product,
            quarter,
            prescriptions,
            prev_prescriptions,
            (prev_prescriptions - prescriptions) AS drop_value
        FROM sales_trend
        WHERE prev_prescriptions IS NOT NULL
          AND prescriptions < prev_prescriptions
        ORDER BY doctor_id, quarter;
        """
        return self.client.execute_query(query, (specialty, product))

    def get_product_performance_by_region(self, product: str):
        query = """
        SELECT
            region,
            product,
            SUM(prescriptions) AS total_prescriptions,
            SUM(sales_value) AS total_sales
        FROM hcp_sales
        WHERE product = %s
        GROUP BY region, product
        ORDER BY total_sales DESC;
        """
        return self.client.execute_query(query, (product,))

    def get_low_engagement_hcps(self, product: str, quarter: str, max_calls: int = 1):
        query = """
        SELECT
            e.doctor_id,
            s.doctor_name,
            s.specialty,
            s.region,
            e.product,
            e.quarter,
            e.total_calls,
            e.email_touches,
            e.speaker_program_attendance,
            e.sample_drops
        FROM rep_engagement_summary e
        LEFT JOIN (
            SELECT DISTINCT doctor_id, doctor_name, specialty, region
            FROM hcp_sales
        ) s
            ON e.doctor_id = s.doctor_id
        WHERE e.product = %s
          AND e.quarter = %s
          AND e.total_calls <= %s
        ORDER BY e.total_calls ASC, e.email_touches ASC;
        """
        return self.client.execute_query(query, (product, quarter, max_calls))

    def get_call_activity_by_doctor(self, doctor_id: str):
        query = """
        SELECT
            doctor_id,
            rep_id,
            product,
            call_date,
            call_type,
            channel,
            call_count,
            discussion_topic
        FROM call_activity
        WHERE doctor_id = %s
        ORDER BY call_date DESC;
        """
        return self.client.execute_query(query, (doctor_id,))

    def get_distinct_doctors(self):
        query = """
        SELECT DISTINCT
            doctor_id,
            doctor_name,
            specialty,
            region
        FROM hcp_sales
        ORDER BY doctor_id;
        """
        return self.client.execute_query(query)


if __name__ == "__main__":
    service = SalesQueryService()

    print("\n=== HCPs with prescription drop ===")
    result_1 = service.get_hcps_with_prescription_drop(
        specialty="Endocrinologist",
        product="GlucoX"
    )
    for row in result_1:
        print(row)

    print("\n=== Product performance by region ===")
    result_2 = service.get_product_performance_by_region(product="GlucoX")
    for row in result_2:
        print(row)

    print("\n=== Low engagement HCPs ===")
    result_3 = service.get_low_engagement_hcps(
        product="GlucoX",
        quarter="2024-Q4",
        max_calls=1
    )
    for row in result_3:
        print(row)