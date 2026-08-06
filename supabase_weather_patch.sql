begin;

-- Ilmaandmed: üks rida ühe kuupäeva kohta.
-- Tulevikus on rida prognoos; pärast päeva möödumist asendatakse see mõõdetud andmetega.
create table if not exists public.weather_daily (
    weather_date date primary key,
    data_kind text not null default 'measured',
    temp_min_c double precision,
    temp_max_c double precision,
    wind_avg_ms double precision,
    radiation_mj_m2 double precision,
    source_station text,
    radiation_station text,
    checked boolean not null default false,
    check_message text,
    updated_at timestamptz not null default now()
);

alter table public.weather_daily add column if not exists data_kind text not null default 'measured';
alter table public.weather_daily add column if not exists temp_min_c double precision;
alter table public.weather_daily add column if not exists temp_max_c double precision;
alter table public.weather_daily add column if not exists wind_avg_ms double precision;
alter table public.weather_daily add column if not exists radiation_mj_m2 double precision;
alter table public.weather_daily add column if not exists source_station text;
alter table public.weather_daily add column if not exists radiation_station text;
alter table public.weather_daily add column if not exists checked boolean not null default false;
alter table public.weather_daily add column if not exists check_message text;
alter table public.weather_daily add column if not exists updated_at timestamptz not null default now();

-- Upsert kasutab weather_date võtit. Lisa unikaalsus ka juhul, kui varasem tabel
-- loodi ilma primary key / unique piiranguta.
create unique index if not exists weather_daily_weather_date_uidx
    on public.weather_daily (weather_date);

create table if not exists public.app_settings (
    key text primary key,
    value text not null,
    updated_at timestamptz not null default now()
);

alter table public.weather_daily enable row level security;
alter table public.app_settings enable row level security;
revoke all on public.weather_daily from anon, authenticated;
revoke all on public.app_settings from anon, authenticated;

commit;
