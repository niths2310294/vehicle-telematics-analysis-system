from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://trip_details_user:IUHbaRAjgGEON0mgdfjiWDRjbYfxBktj@dpg-d75st094tr6s73ce0hd0-a.singapore-postgres.render.com:5432/trip_details"

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM trips"))

    for row in result:
        print(row)