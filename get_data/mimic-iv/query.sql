with times as (
    select subject_id, hadm_id, charttime
    from`physionet-data.mimic_derived.bg` bg
    union distinct
    select subject_id, hadm_id, charttime
    from `physionet-data.mimic_derived.blood_differential`
    union distinct
    select subject_id, hadm_id, charttime
    from `physionet-data.mimic_derived.cardiac_marker`
    union distinct
    select subject_id, hadm_id, charttime
    from `physionet-data.mimic_derived.chemistry`
    union distinct
    select subject_id, hadm_id, charttime
    from `physionet-data.mimic_derived.coagulation`
    union distinct
    select subject_id, hadm_id, charttime
    from `physionet-data.mimic_derived.complete_blood_count`
    union distinct
    select subject_id, hadm_id, charttime
    from `physionet-data.mimic_derived.enzyme`
)

, all_labs as (
    select times.*
            , bg.specimen, bg.specimen_pred, bg.specimen_prob, bg.so2, bg.po2, bg.pco2, bg.fio2_chartevents, bg.fio2, bg.aado2, bg.aado2_calc
            , bg.pao2fio2ratio, bg.ph, bg.baseexcess, bg.bicarbonate as bicarbonate_bg, bg.totalco2, bg.hematocrit as hematocrit_bg, bg.hemoglobin as hemoglobin_bg, bg.carboxyhemoglobin
            , bg.methemoglobin, bg.chloride as chloride_bg, bg.calcium as calcium_bg, bg.temperature, bg.potassium as potassium_bg, bg.sodium as sodium_bg
            , bg.lactate, bg.glucose as glucose_bg
            , blood.wbc, blood.basophils_abs, blood.eosinophils_abs, blood.lymphocytes_abs, blood.monocytes_abs, blood.neutrophils_abs
            , blood.basophils, blood.eosinophils, blood.lymphocytes, blood.monocytes, blood.neutrophils, blood.atypical_lymphocytes, blood.bands
            , blood.immature_granulocytes, blood.metamyelocytes, blood.nrbc
            , card.troponin_i, card.troponin_t--, card.ck_mb duplicated below
            , enz.alt, enz.alp, enz.ast, enz.amylase, enz.bilirubin_total, enz.bilirubin_direct, enz.bilirubin_indirect, enz.ck_cpk, enz.ck_mb, enz.ggt, enz.ld_ldh
            , chem.albumin, chem.globulin, chem.total_protein, chem.aniongap, chem.bicarbonate, chem.bun, chem.calcium, chem.chloride, chem.creatinine, chem.glucose
            , chem.sodium, chem.potassium
            , coag.d_dimer, coag.fibrinogen, coag.thrombin, coag.inr, coag.pt, coag.ptt
            , cbc.hematocrit, cbc.hemoglobin, cbc.mch, cbc.mchc, cbc.mcv, cbc.platelet, cbc.rbc, cbc.rdw, cbc.rdwsd--, cbc.wbc, a subset of the previous wbc

    from times
    left join `physionet-data.mimic_derived.bg` bg
    on times.subject_id = bg.subject_id
    and times.charttime = bg.charttime
    left join `physionet-data.mimic_derived.blood_differential` blood
    on times.subject_id = blood.subject_id
    and times.charttime = blood.charttime
    left join `physionet-data.mimic_derived.cardiac_marker` card
    on times.subject_id = card.subject_id
    and times.charttime = card.charttime
    left join `physionet-data.mimic_derived.chemistry` chem
    on times.subject_id = chem.subject_id
    and times.charttime = chem.charttime
    left join `physionet-data.mimic_derived.coagulation` coag
    on times.subject_id = coag.subject_id
    and times.charttime = coag.charttime
    left join `physionet-data.mimic_derived.complete_blood_count` cbc
    on times.subject_id = cbc.subject_id
    and times.charttime = cbc.charttime
    left join `physionet-data.mimic_derived.enzyme` enz
    on times.subject_id = enz.subject_id
    and times.charttime = enz.charttime
    --where times.subject_id in (13978368, 15296256, 18771968)
    order by times.subject_id, times.charttime
)

-- Note in the following we are only loosely joining, with more time, should write a script to group all labs with the relevant stays like in MIMICIII
, labs_icu as (
    select ic.stay_id, lab.*
    --from all_labs
    from `physionet-data.mimic_icu.icustays` ic
    inner join all_labs lab
    on ic.subject_id = lab.subject_id
    and lab.charttime >= datetime_sub(ic.intime, interval '6' hour)
    and lab.charttime <= datetime_add(ic.outtime, interval '6' hour)
)

, vitalsigns as (
    select
        ce.subject_id
      , ce.stay_id
      , ce.charttime
      --, MAX(ce.storetime)
      , AVG(case when itemid in (220045) and valuenum > 0 and valuenum < 300 then valuenum else null end) as heart_rate
      , AVG(case when itemid in (225309,220050) and valuenum > 0 and valuenum < 400 then valuenum else null end) as sbp
      , AVG(case when itemid in (225310,220051) and valuenum > 0 and valuenum < 300 then valuenum else null end) as dbp
      , AVG(case when itemid in (220052,220181,225312) and valuenum > 0 and valuenum < 300 then valuenum else null end) as mbp
      , AVG(case when itemid = 220179 and valuenum > 0 and valuenum < 400 then valuenum else null end) as sbp_ni
      , AVG(case when itemid = 220180 and valuenum > 0 and valuenum < 300 then valuenum else null end) as dbp_ni
      , AVG(case when itemid = 220181 and valuenum > 0 and valuenum < 300 then valuenum else null end) as mbp_ni
      , AVG(case when itemid in (220210,224690) and valuenum > 0 and valuenum < 70 then valuenum else null end) as resp_rate
      , ROUND(
          AVG(case when itemid in (223761) and valuenum > 70 and valuenum < 120 then (valuenum-32)/1.8 -- converted to degC in valuenum call
                  when itemid in (223762) and valuenum > 10 and valuenum < 50  then valuenum else null end)
        , 2) as temperature
      , MAX(CASE WHEN itemid = 224642 THEN value ELSE NULL END) AS temperature_site
      , AVG(case when itemid in (220277) and valuenum > 0 and valuenum <= 100 then valuenum else null end) as spo2
      , AVG(case when itemid in (225664,220621,226537) and valuenum > 0 then valuenum else null end) as glucose
      FROM `physionet-data.mimic_icu.chartevents` ce
      where ce.stay_id IS NOT NULL
      and ce.itemid in
      (
        220045, -- Heart Rate
        225309, -- ART BP Systolic
        225310, -- ART BP Diastolic
        225312, -- ART BP Mean
        220050, -- Arterial Blood Pressure systolic
        220051, -- Arterial Blood Pressure diastolic
        220052, -- Arterial Blood Pressure mean
        220179, -- Non Invasive Blood Pressure systolic
        220180, -- Non Invasive Blood Pressure diastolic
        220181, -- Non Invasive Blood Pressure mean
        220210, -- Respiratory Rate
        224690, -- Respiratory Rate (Total)
        220277, -- SPO2, peripheral
        -- GLUCOSE, both lab and fingerstick
        225664, -- Glucose finger stick
        220621, -- Glucose (serum)
        226537, -- Glucose (whole blood)
        -- TEMPERATURE
        223762, -- "Temperature Celsius"
        223761,  -- "Temperature Fahrenheit"
        224642 -- Temperature Site
        -- 226329 -- Blood Temperature CCO (C)
    )
    group by ce.subject_id, ce.stay_id, ce.charttime
    --order by ce.subject_id, ce.stay_id, ce.charttime

)

, times2 as (
    select subject_id, stay_id, charttime
    from vitalsigns --`physionet-data.mimic_derived.vitalsign`
    union distinct
    select subject_id, stay_id, charttime
    from `physionet-data.mimic_derived.ventilator_setting`
)

, charts as (
    select times2.*, vit.heart_rate, vit.sbp, vit.dbp, vit.mbp, vit.sbp_ni, vit.dbp_ni, vit.mbp_ni, vit.resp_rate, vit.temperature, vit.temperature_site
        , vit.spo2, vit.glucose
        , vent.respiratory_rate_set, vent.respiratory_rate_total, vent.respiratory_rate_spontaneous, vent.minute_volume, vent.tidal_volume_set, vent.tidal_volume_observed
        , vent.tidal_volume_spontaneous, vent.plateau_pressure, vent.peep, vent.fio2, vent.ventilator_mode, vent.ventilator_mode_hamilton, vent.ventilator_type
    from times2
    left join vitalsigns vit
    on times2.stay_id = vit.stay_id
    and times2.charttime = vit.charttime
    left join `physionet-data.mimic_derived.ventilator_setting` vent
    on times2.stay_id = vent.stay_id
    and times2.charttime = vent.charttime
    order by times2.subject_id, times2.stay_id, times2.charttime
)

, times3 as (
    select subject_id, stay_id, charttime
    from charts
    union distinct
    select subject_id, stay_id, charttime
    from labs_icu
    union distinct
    select subject_id, stay_id, charttime
    from `physionet-data.mimic_derived.oxygen_delivery`
)

, combined as (
    select times3.*
            , charts.heart_rate, charts.sbp, charts.dbp, charts.mbp, charts.sbp_ni, charts.dbp_ni, charts.mbp_ni, charts.resp_rate, charts.temperature, charts.temperature_site
            , charts.spo2, charts.glucose, charts.respiratory_rate_set, charts.respiratory_rate_total, charts.respiratory_rate_spontaneous, charts.minute_volume
            , charts.tidal_volume_set, charts.tidal_volume_observed, charts.tidal_volume_spontaneous, charts.plateau_pressure, charts.peep, charts.fio2, charts.ventilator_mode
            , charts.ventilator_mode_hamilton, charts.ventilator_type
            , labs.specimen, labs.specimen_pred, labs.specimen_prob, labs.so2, labs.po2, labs.pco2, labs.fio2 as fio2_labs, labs.aado2, labs.aado2_calc --labs.fio2_chartevents a repeat of charts.fio2
            , labs.pao2fio2ratio, labs.ph, labs.baseexcess, labs.bicarbonate_bg, labs.totalco2, labs.hematocrit_bg, labs.hemoglobin_bg, labs.carboxyhemoglobin
            , labs.methemoglobin, labs.chloride_bg, labs.calcium_bg, labs.temperature as temperature_labs, labs.potassium_bg, labs.sodium_bg
            , labs.lactate, labs.glucose_bg
            , labs.wbc, labs.basophils_abs, labs.eosinophils_abs, labs.lymphocytes_abs, labs.monocytes_abs, labs.neutrophils_abs
            , labs.basophils, labs.eosinophils, labs.lymphocytes, labs.monocytes, labs.neutrophils, labs.atypical_lymphocytes, labs.bands
            , labs.immature_granulocytes, labs.metamyelocytes, labs.nrbc
            , labs.troponin_i, labs.troponin_t
            , labs.alt, labs.alp, labs.ast, labs.amylase, labs.bilirubin_total, labs.bilirubin_direct, labs.bilirubin_indirect, labs.ck_cpk, labs.ck_mb, labs.ggt, labs.ld_ldh
            , labs.albumin, labs.globulin, labs.total_protein, labs.aniongap, labs.bicarbonate, labs.bun, labs.calcium, labs.chloride, labs.creatinine, labs.glucose as glucose_labs
            , labs.sodium, labs.potassium
            , labs.d_dimer, labs.fibrinogen, labs.thrombin, labs.inr, labs.pt, labs.ptt
            , labs.hematocrit, labs.hemoglobin, labs.mch, labs.mchc, labs.mcv, labs.platelet, labs.rbc, labs.rdw, labs.rdwsd
            , vent.ventilation_status
            , o2.o2_flow, o2.o2_delivery_device_1 as o2_delivery_device

    from times3
    left join charts
    on times3.stay_id = charts.stay_id
    and times3.charttime = charts.charttime
    left join labs_icu labs
    on times3.stay_id = labs.stay_id
    and times3.charttime = labs.charttime
    left join `physionet-data.mimic_derived.oxygen_delivery` o2
    on times3.stay_id = o2.stay_id
    and times3.charttime = o2.charttime
    left join `physionet-data.mimic_derived.ventilation` vent
    on times3.stay_id = vent.stay_id
    and times3.charttime >= vent.starttime
    and times3.charttime < vent.endtime --or vent.starttime is null
)

-- The following temp tables are for sepsis definition as we have in the previous paper. An alternative can be found in `physionet-data.mimic_derived.sepsis3`, they use absolute sofa rather than a change of 2.
-- Compared with MIMICIII, this may not do much to identify SOFA on the wards, this can be put on the to-do list.

-- Note this query does not do much in the way of finding suspicion of infections before ICU admission, in that case the prescriptions table should be incorporated

-- We explore the idea of splitting medications by different treatment cycles and removing patients with isolated antibiotic events

-- As mentioned in the sepsis-3 paper, isolated cases of antibiotics would not satisfy the requirement for a suspicion of infection. We need another dose of antibiotics within 96 hours of the first one

, abx_partition as
(
select *,
 sum( new_antibiotics_cycle )
      over ( partition by stay_id, antibiotic_name order by stay_id, starttime )
    as abx_num
 from
  (select subject_id, hadm_id, stay_id, starttime, endtime
   , label as antibiotic_name
   -- Check if the same antibiotics was taken in the last 4 days
   , case when starttime <= datetime_add(lag(endtime) over (partition by stay_id, label order by stay_id, starttime), interval 96 hour) then 0 else 1 end as new_antibiotics_cycle
   , datetime_diff(lead(starttime) over (partition by stay_id order by stay_id, starttime), endtime, second)/3600 as next_antibio_time
   --, datetime_diff(lead(starttime) over (partition by stay_id order by stay_id, starttime), endtime, hour) as next_antibio_time
   , case when lead(starttime) over (partition by stay_id order by stay_id, starttime)  <= datetime_add(endtime, interval 96 hour) then 0 else 1 end as isolated_case
   from `physionet-data.mimic_icu.inputevents` input
inner join `physionet-data.mimic_icu.d_items` d on input.itemid = d.itemid
where upper(d.category) like '%ANTIBIOTICS%'
) A
order by subject_id, hadm_id, stay_id, starttime
)

-- group the antibiotic information together to form courses of antibiotics and also to check whether they are isolated cases

-- note the last drug dose taken by the patient will be classed as an isolated case. However, if this is of the same type as antibiotics given to patient within last 4 days, then it will be aggreagated with the other doses in the next query.

, abx_partition_grouped as
(
select subject_id, hadm_id, stay_id, min(starttime) as starttime, max(endtime) as endtime
    , count(*) as doses, antibiotic_name, min(isolated_case) as isolated_case
from abx_partition
group by subject_id, hadm_id, stay_id, antibiotic_name, abx_num
order by subject_id, hadm_id, stay_id, starttime
)

, ab_tbl as
(
  select
        ie.subject_id, ie.hadm_id, ie.stay_id
      , ie.intime, ie.outtime
      , case when ab.isolated_case = 0 then ab.antibiotic_name else null end as antibiotic_name
      , case when ab.isolated_case = 0 then ab.starttime else null end as antibiotic_time
      --, ab.isolated_case
      --, abx.endtime
  from `physionet-data.mimic_icu.icustays` ie
  left join abx_partition_grouped ab
      on ie.hadm_id = ab.hadm_id and ie.stay_id = ab.stay_id
)

-- Find the microbiology events
, me as
(
  select subject_id, hadm_id
    , chartdate, charttime
    , spec_type_desc
    , max(case when org_name is not null and org_name != '' then 1 else 0 end) as PositiveCulture
  from  `physionet-data.mimic_hosp.microbiologyevents`
  group by subject_id, hadm_id, chartdate, charttime, spec_type_desc
)

, ab_fnl as
(
  select
      ab_tbl.stay_id, ab_tbl.intime, ab_tbl.outtime
    , ab_tbl.antibiotic_name
    , ab_tbl.antibiotic_time
    , coalesce(me.charttime,me.chartdate) as culture_charttime
    , me.positiveculture as positiveculture
    , me.spec_type_desc as specimen
    , case
      when coalesce(antibiotic_time,coalesce(me.charttime,me.chartdate)) is null
        then 0
      else 1 end as suspected_infection
    , least(antibiotic_time, coalesce(me.charttime,me.chartdate)) as t_suspicion
  from ab_tbl
  left join me
    on ab_tbl.hadm_id = me.hadm_id
    and ab_tbl.antibiotic_time is not null
    and
    (
      -- if charttime is available, use it
      (
          ab_tbl.antibiotic_time >= datetime_sub(me.charttime, interval 24 hour)
      and ab_tbl.antibiotic_time <= datetime_add(me.charttime, interval 72 hour)

      )
      OR
      (
      -- if charttime is not available, use chartdate
          me.charttime is null
      and ab_tbl.antibiotic_time >= datetime_sub(me.chartdate, interval 24 hour)
      and ab_tbl.antibiotic_time < datetime_add(me.chartdate, interval 96 hour) --  Note this is 96 hours to include cases of when antibiotics are given 3 days after chart date of culture
      )
    )
)

-- select only the unique times for suspicion of infection
, unique_times as
(
select stay_id, t_suspicion, count(*) as repeats from ab_fnl
group by stay_id, t_suspicion
order by stay_id, t_suspicion
)


-- Around each suspicion of infection, check the changes of the current SOFA score from the beginning of the window
, sofa_scores as
(
select
    sus.stay_id, hr, starttime, endtime, t_suspicion, sofa_24hours
    , first_value(sofa_24hours) over (partition by sofa.stay_id, t_suspicion order by sofa.stay_id, t_suspicion, starttime) as initial_sofa
    , sofa_24hours - first_value(sofa_24hours) over (partition by sofa.stay_id, t_suspicion order by sofa.stay_id, t_suspicion, starttime) as sofa_difference
from unique_times sus --`physionet-data.mimic_derived.suspicion_of_infection` sus
left join `physionet-data.mimic_derived.sofa` sofa
    on sus.stay_id = sofa.stay_id
-- the following is the largest interval suggested by the sepsis-3 paper, though their sensitivity analysis checked +-3 hours upwards
where starttime <= datetime_add(t_suspicion, interval 24 hour)
    and starttime >= datetime_sub(t_suspicion, interval 48 hour)
order by sofa.stay_id, t_suspicion, starttime
)

-- find where the SOFA score has increased by 2 within the time sensitivity being investigated
-- Note the sepsis-3 papers mentions that the baseline of a patient should be zero so an alternative is just to test if the total SOFA score exceeds 2. However that approach would have lower specificity and may be less clinically interesting
, sofa_times as
(
select stay_id, t_suspicion, min(starttime) as t_sofa
from sofa_scores
where sofa_difference>=2
group by stay_id, t_suspicion
)

-- Find the first time when the sepsis-3 requirements are satisfied
, first_event as
(
select stay_id, min(t_suspicion) as t_suspicion, min(t_sofa) as t_sofa from sofa_times
    group by stay_id
)

, sepsis_times as (
  select subject_id, hadm_id, ie.stay_id--, intime, outtime
    , t_suspicion, t_sofa
    -- some papers may take the minimum to define sepsis time
    , least(t_suspicion, t_sofa) as t_sepsis_min
  from `physionet-data.mimic_icu.icustays` ie
  left join first_event fe on ie.stay_id = fe.stay_id
  order by subject_id, intime
)

, order_weight as (
    select * --stay_id, weight
    , ROW_NUMBER() OVER(PARTITION BY stay_id order by starttime) as rn
    from `physionet-data.mimic_derived.weight_durations`
    where weight < 350 and weight > 25

)

, weight_height as (
    select od.stay_id, height, weight
    from `physionet-data.mimic_derived.height` he
    full outer join order_weight od
    on he.stay_id = od.stay_id
    where rn = 1
)

, icu_info as (
    select ic.*, wh.height, wh.weight, sep.t_suspicion, sep.t_sofa, sep.t_sepsis_min
    from `physionet-data.mimic_derived.icustay_detail` ic
    inner join sepsis_times sep
    on ic.stay_id = sep.stay_id
    left join weight_height wh
    on ic.stay_id = wh.stay_id
)

, merged_table as (
    select icu.*
            , comb.charttime
            , comb.heart_rate, comb.sbp, comb.dbp, comb.mbp, comb.sbp_ni, comb.dbp_ni, comb.mbp_ni, comb.resp_rate, comb.temperature, comb.temperature_site
            , comb.spo2, comb.glucose, comb.respiratory_rate_set, comb.respiratory_rate_total, comb.respiratory_rate_spontaneous, comb.minute_volume
            , comb.tidal_volume_set, comb.tidal_volume_observed, comb.tidal_volume_spontaneous, comb.plateau_pressure, comb.peep, comb.fio2, comb.ventilator_mode
            , comb.ventilator_mode_hamilton, comb.ventilator_type
            , comb.specimen, comb.specimen_pred, comb.specimen_prob, comb.so2, comb.po2, comb.pco2, comb.fio2_labs, comb.aado2, comb.aado2_calc
            , comb.pao2fio2ratio, comb.ph, comb.baseexcess, comb.bicarbonate_bg, comb.totalco2, comb.hematocrit_bg, comb.hemoglobin_bg, comb.carboxyhemoglobin
            , comb.methemoglobin, comb.chloride_bg, comb.calcium_bg, comb.temperature_labs, comb.potassium_bg, comb.sodium_bg
            , comb.lactate, comb.glucose_bg
            , comb.wbc, comb.basophils_abs, comb.eosinophils_abs, comb.lymphocytes_abs, comb.monocytes_abs, comb.neutrophils_abs
            , comb.basophils, comb.eosinophils, comb.lymphocytes, comb.monocytes, comb.neutrophils, comb.atypical_lymphocytes, comb.bands
            , comb.immature_granulocytes, comb.metamyelocytes, comb.nrbc
            , comb.troponin_i, comb.troponin_t
            , comb.alt, comb.alp, comb.ast, comb.amylase, comb.bilirubin_total, comb.bilirubin_direct, comb.bilirubin_indirect, comb.ck_cpk, comb.ck_mb, comb.ggt, comb.ld_ldh
            , comb.albumin, comb.globulin, comb.total_protein, comb.aniongap, comb.bicarbonate, comb.bun, comb.calcium, comb.chloride, comb.creatinine, comb.glucose_labs
            , comb.sodium, comb.potassium
            , comb.d_dimer, comb.fibrinogen, comb.thrombin, comb.inr, comb.pt, comb.ptt
            , comb.hematocrit, comb.hemoglobin, comb.mch, comb.mchc, comb.mcv, comb.platelet, comb.rbc, comb.rdw, comb.rdwsd
            , comb.ventilation_status
            , comb.o2_flow, comb.o2_delivery_device


    from icu_info icu
    inner join combined comb
    on icu.stay_id = comb.stay_id
    -- too expensive to order here
    --order by icu.subject_id, icu.hadm_id, icu.stay_id, comb.charttime

)

select *
from merged_table
