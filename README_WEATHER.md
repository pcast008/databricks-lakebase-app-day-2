# Weather Intelligence — Homework Notes

## Which data source you chose and why

I chose the NWS because it was the source suggested in the homework and I
needed to move quickly. 

## Your schema decisions (columns, chunking parameters, embedding model/dimensions).

I mirrored the news schema, but also added a `source_type` column to support multi-source and satisfy the extra credit.

## How to run the pipeline end-to-end

First run the DDL scripts to create all the weather tables. Then run the ingest_weather_embeddings to load the embeddings. Finally, to test the search, run the search_weather notebook.

## Known limitations / what I'd improve with more time

- I'd build the full frontend UI to see the weather data and search in action.
- I'd spend more time tuning the chunking and digesting the ingest notebook to
  really understand what's happening.
