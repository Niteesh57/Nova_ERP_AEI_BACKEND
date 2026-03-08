import sqlite3

conn = sqlite3.connect("db/nova_aei.db")
c = conn.cursor()
c.execute("UPDATE leads SET contact_status = 'uncontacted' WHERE contact_status IS NULL OR contact_status = ''")
conn.commit()
print("Updated", c.rowcount, "rows")
conn.close()
