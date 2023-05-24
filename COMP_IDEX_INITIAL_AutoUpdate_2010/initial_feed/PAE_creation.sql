drop table GKP_PAE_6hr;
drop table GKP_PAE_48hr;
drop table SNG_PAE_6hr;
drop table SNG_PAE_48hr;
drop table HND_PAE_6hr;
drop table HND_PAE_48hr;
drop table CDH_PAE_6hr;
drop table CDH_PAE_48hr;


create table GKP_PAE_6hr (datetime TIMESTAMP, "pred(t-6.0)" float, "error(t-6.0)" float, "pred(t-5.5)" float, "error(t-5.5)" float,
"pred(t-5.0)" float, "error(t-5.0)" float, "pred(t-4.5)" float, "error(t-4.5)" float, "pred(t-4.0)" float, "error(t-4.0)" float,
"pred(t-3.5)" float, "error(t-3.5)" float, "pred(t-3.0)" float, "error(t-3.0)" float, "pred(t-2.5)" float, "error(t-2.5)" float,
"pred(t-2.0)" float, "error(t-2.0)" float,"pred(t-1.5)" float, "error(t-1.5)" float,"pred(t-1.0)" float, "error(t-1.0)" float,
"pred(t-0.5)" float, "error(t-0.5)" float, "actual(t)" float);


create table GKP_PAE_48hr (datetime TIMESTAMP,
"pred(t-48)" float, "error(t-48)" float, "pred(t-46)" float, "error(t-46)" float, "pred(t-44)" float, "error(t-44)" float,
"pred(t-42)" float, "error(t-42)" float, "pred(t-40)" float, "error(t-40)" float, "pred(t-38)" float, "error(t-38)" float, 
"pred(t-36)" float, "error(t-36)" float, "pred(t-34)" float, "error(t-34)" float, "pred(t-32)" float, "error(t-32)" float, 
"pred(t-30)" float, "error(t-30)" float, "pred(t-28)" float, "error(t-28)" float, "pred(t-26)" float, "error(t-26)" float, 
"pred(t-24)" float, "error(t-24)" float, "pred(t-22)" float, "error(t-22)" float, "pred(t-20)" float, "error(t-20)" float, 
"pred(t-18)" float, "error(t-18)" float, "pred(t-16)" float, "error(t-16)" float, "pred(t-14)" float, "error(t-14)" float, 
"pred(t-12)" float, "error(t-12)" float, "pred(t-10)" float, "error(t-10)" float, "pred(t-8)" float,  "error(t-8)" float,
"pred(t-6)" float,  "error(t-6)" float,  "pred(t-4)" float,  "error(t-4)" float,  "pred(t-2)" float,  "error(t-2)" float,
"actual(t)" float);



create table SNG_PAE_6hr (datetime TIMESTAMP, "pred(t-6.0)" float, "error(t-6.0)" float, "pred(t-5.5)" float, "error(t-5.5)" float,
"pred(t-5.0)" float, "error(t-5.0)" float, "pred(t-4.5)" float, "error(t-4.5)" float, "pred(t-4.0)" float, "error(t-4.0)" float,
"pred(t-3.5)" float, "error(t-3.5)" float, "pred(t-3.0)" float, "error(t-3.0)" float, "pred(t-2.5)" float, "error(t-2.5)" float,
"pred(t-2.0)" float, "error(t-2.0)" float,"pred(t-1.5)" float, "error(t-1.5)" float,"pred(t-1.0)" float, "error(t-1.0)" float,
"pred(t-0.5)" float, "error(t-0.5)" float, "actual(t)" float);


create table SNG_PAE_48hr (datetime TIMESTAMP,
"pred(t-48)" float, "error(t-48)" float, "pred(t-46)" float, "error(t-46)" float, "pred(t-44)" float, "error(t-44)" float,
"pred(t-42)" float, "error(t-42)" float, "pred(t-40)" float, "error(t-40)" float, "pred(t-38)" float, "error(t-38)" float, 
"pred(t-36)" float, "error(t-36)" float, "pred(t-34)" float, "error(t-34)" float, "pred(t-32)" float, "error(t-32)" float, 
"pred(t-30)" float, "error(t-30)" float, "pred(t-28)" float, "error(t-28)" float, "pred(t-26)" float, "error(t-26)" float, 
"pred(t-24)" float, "error(t-24)" float, "pred(t-22)" float, "error(t-22)" float, "pred(t-20)" float, "error(t-20)" float, 
"pred(t-18)" float, "error(t-18)" float, "pred(t-16)" float, "error(t-16)" float, "pred(t-14)" float, "error(t-14)" float, 
"pred(t-12)" float, "error(t-12)" float, "pred(t-10)" float, "error(t-10)" float, "pred(t-8)" float,  "error(t-8)" float,
"pred(t-6)" float,  "error(t-6)" float,  "pred(t-4)" float,  "error(t-4)" float,  "pred(t-2)" float,  "error(t-2)" float,
"actual(t)" float);




create table HND_PAE_6hr (datetime TIMESTAMP, "pred(t-6.0)" float, "error(t-6.0)" float, "pred(t-5.5)" float, "error(t-5.5)" float,
"pred(t-5.0)" float, "error(t-5.0)" float, "pred(t-4.5)" float, "error(t-4.5)" float, "pred(t-4.0)" float, "error(t-4.0)" float,
"pred(t-3.5)" float, "error(t-3.5)" float, "pred(t-3.0)" float, "error(t-3.0)" float, "pred(t-2.5)" float, "error(t-2.5)" float,
"pred(t-2.0)" float, "error(t-2.0)" float,"pred(t-1.5)" float, "error(t-1.5)" float,"pred(t-1.0)" float, "error(t-1.0)" float,
"pred(t-0.5)" float, "error(t-0.5)" float, "actual(t)" float);


create table HND_PAE_48hr (datetime TIMESTAMP,
"pred(t-48)" float, "error(t-48)" float, "pred(t-46)" float, "error(t-46)" float, "pred(t-44)" float, "error(t-44)" float,
"pred(t-42)" float, "error(t-42)" float, "pred(t-40)" float, "error(t-40)" float, "pred(t-38)" float, "error(t-38)" float, 
"pred(t-36)" float, "error(t-36)" float, "pred(t-34)" float, "error(t-34)" float, "pred(t-32)" float, "error(t-32)" float, 
"pred(t-30)" float, "error(t-30)" float, "pred(t-28)" float, "error(t-28)" float, "pred(t-26)" float, "error(t-26)" float, 
"pred(t-24)" float, "error(t-24)" float, "pred(t-22)" float, "error(t-22)" float, "pred(t-20)" float, "error(t-20)" float, 
"pred(t-18)" float, "error(t-18)" float, "pred(t-16)" float, "error(t-16)" float, "pred(t-14)" float, "error(t-14)" float, 
"pred(t-12)" float, "error(t-12)" float, "pred(t-10)" float, "error(t-10)" float, "pred(t-8)" float,  "error(t-8)" float,
"pred(t-6)" float,  "error(t-6)" float,  "pred(t-4)" float,  "error(t-4)" float,  "pred(t-2)" float,  "error(t-2)" float,
"actual(t)" float);


create table CDH_PAE_6hr (datetime TIMESTAMP, "pred(t-6.0)" float, "error(t-6.0)" float, "pred(t-5.5)" float, "error(t-5.5)" float,
"pred(t-5.0)" float, "error(t-5.0)" float, "pred(t-4.5)" float, "error(t-4.5)" float, "pred(t-4.0)" float, "error(t-4.0)" float,
"pred(t-3.5)" float, "error(t-3.5)" float, "pred(t-3.0)" float, "error(t-3.0)" float, "pred(t-2.5)" float, "error(t-2.5)" float,
"pred(t-2.0)" float, "error(t-2.0)" float,"pred(t-1.5)" float, "error(t-1.5)" float,"pred(t-1.0)" float, "error(t-1.0)" float,
"pred(t-0.5)" float, "error(t-0.5)" float, "actual(t)" float);

create table CDH_PAE_48hr (datetime TIMESTAMP,
"pred(t-48)" float, "error(t-48)" float, "pred(t-46)" float, "error(t-46)" float, "pred(t-44)" float, "error(t-44)" float,
"pred(t-42)" float, "error(t-42)" float, "pred(t-40)" float, "error(t-40)" float, "pred(t-38)" float, "error(t-38)" float, 
"pred(t-36)" float, "error(t-36)" float, "pred(t-34)" float, "error(t-34)" float, "pred(t-32)" float, "error(t-32)" float, 
"pred(t-30)" float, "error(t-30)" float, "pred(t-28)" float, "error(t-28)" float, "pred(t-26)" float, "error(t-26)" float, 
"pred(t-24)" float, "error(t-24)" float, "pred(t-22)" float, "error(t-22)" float, "pred(t-20)" float, "error(t-20)" float, 
"pred(t-18)" float, "error(t-18)" float, "pred(t-16)" float, "error(t-16)" float, "pred(t-14)" float, "error(t-14)" float, 
"pred(t-12)" float, "error(t-12)" float, "pred(t-10)" float, "error(t-10)" float, "pred(t-8)" float,  "error(t-8)" float,
"pred(t-6)" float,  "error(t-6)" float,  "pred(t-4)" float,  "error(t-4)" float,  "pred(t-2)" float,  "error(t-2)" float,
"actual(t)" float);

