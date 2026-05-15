from engine.data import duckdb_store
con = duckdb_store.connect(read_only=True)
r = con.execute("SELECT DISTINCT symbol, COUNT(*) as cnt FROM bars WHERE timeframe='M1' GROUP BY symbol ORDER BY symbol").fetchdf()
print(r.to_string())
con.close()
