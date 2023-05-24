drop table cities;

create table cities(city_id int, city_name varchar(50), begin_datetime timestamp);
commit;

insert into cities values(1,'Srinagar', to_date('2018-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') );
insert into cities values(2,'Gorakhpur', to_date('2018-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS'));
insert into cities values(3,'Hindan',to_date('2018-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS'));
insert into cities values(4,'Chandigarh', to_date('2018-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS'));
commit;

drop table forecast;

create table forecast(forecast_hours varchar(50));
commit;

insert into forecast values('6hr');
commit;
insert into forecast values('48hr');
commit;




drop table countdb;
create table countdb (v int);

insert into countdb values(0);
update countdb set v=0;
commit;