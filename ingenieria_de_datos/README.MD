# Data Engineering summary
**Responsible: ** Nicolas Salazar Sanchez

## Data process
1. Get data from: https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe/about_data 
We get the data from the official Icfes/Saber source. In particular, the date the CSV file was extracted was 17/02/2026. It contains about 7.1MM rows.
2. Create AWS instances to help process data. 
- S3 buket's: proy1-entry-bucket, where the CSV was stored, and proy1-exit-bucket, where the athena queries and the glue crawlers search will be stored. 
- Athena service: proy1-workgroup -> analytic engine: Athena SQL -> query result: proy1-exit-bucket -> data source AWS Glue data Catalog.
- AWS Glue Data Catalog with it's associate crawler, crawling the Icfes CSV file. 
After this, we just run the crawler so we can make the Athena SQL queries later on. 
