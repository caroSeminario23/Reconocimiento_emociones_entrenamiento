
CREATE TABLE Detalle_resultado
(
  id_res_sesion  INTEGER NOT NULL,
  id_estado_sm   INTEGER NOT NULL,
  -- de la sesión
  por_frecuencia REAL    NOT NULL,
  PRIMARY KEY (id_res_sesion, id_estado_sm),
  FOREIGN KEY (id_estado_sm) REFERENCES Estado_sm (id_estado_sm),
  FOREIGN KEY (id_res_sesion) REFERENCES Resultado_sesion (id_res_sesion)
);

-- salud mental
CREATE TABLE Estado_sm
(
  id_estado_sm INTEGER NOT NULL,
  nombre       TEXT    NOT NULL,
  -- precision de 3 decimales
  valor_min    REAL    NOT NULL,
  -- precision de 3 decimales
  valor_max    REAL    NOT NULL,
  PRIMARY KEY (id_estado_sm)
);

-- por minuto
CREATE TABLE Lote_fotogramas
(
  id_lote         INTEGER NOT NULL,
  n_fotogramas    INTEGER NOT NULL,
  -- precision de 3 decimales
  valor_min_pred  REAL    NOT NULL,
  -- precision de 3 decimales
  valor_max_pred  REAL    NOT NULL,
  -- precision de 3 decimales
  valor_prom_pred REAL    NOT NULL,
  id_estado_sm    INTEGER NOT NULL,
  -- (YYYY-MM-DD)
  fecha           TEXT    NOT NULL,
  -- minuto inicial (HH:MM:SS)
  hora_inicio     TEXT    NOT NULL,
  -- minuto final (HH:MM:SS)
  hora_fin        TEXT    NOT NULL,
  -- fps_estado_no_frecuente/fps_total_estados
  prob_error_pred REAL    NOT NULL,
  id_res_sesion   INTEGER NOT NULL,
  PRIMARY KEY (id_lote),
  FOREIGN KEY (id_estado_sm) REFERENCES Estado_sm (id_estado_sm),
  FOREIGN KEY (id_res_sesion) REFERENCES Resultado_sesion (id_res_sesion)
);

CREATE TABLE Paciente
(
  id_paciente INTEGER NOT NULL,
  PRIMARY KEY (id_paciente)
);

CREATE TABLE Respuesta_IA
(
  id_rpt        INTEGER NOT NULL,
  id_res_sesion INTEGER NOT NULL,
  -- 0: no, 1: si
  respondio     INTEGER NOT NULL,
  -- de la intervencion (HH:MM:SS)
  duracion      TEXT    NOT NULL,
  -- resumen de la intervención
  intervencion  TEXT    NOT NULL,
  PRIMARY KEY (id_rpt),
  FOREIGN KEY (id_res_sesion) REFERENCES Resultado_sesion (id_res_sesion)
);

CREATE TABLE Resultado_sesion
(
  id_res_sesion INTEGER NOT NULL,
  id_paciente   INTEGER NOT NULL,
  -- (YYYY-MM-DD)
  fecha         TEXT    NOT NULL,
  -- (HH:MM:SS)
  hora_inicio   TEXT    NOT NULL,
  -- (HH:MM:SS)
  hora_fin      TEXT    NOT NULL,
  PRIMARY KEY (id_res_sesion),
  FOREIGN KEY (id_paciente) REFERENCES Paciente (id_paciente)
);

-- tabla temporal o en un archivo csv temporal
CREATE TABLE Resultados_temp
(
  id_res_temp      INTEGER NOT NULL,
  -- precision de 3 decimales
  valor_prediccion REAL    NOT NULL,
  -- (YYYY-MM-DD)
  fecha            TEXT    NOT NULL,
  -- (HH:MM:SS)
  hora             TEXT    NOT NULL,
  PRIMARY KEY (id_res_temp)
);
