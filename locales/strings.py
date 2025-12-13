#!/usr/bin/env python3
# GPSDyno - GPS-based vehicle power calculator
# Copyright (C) 2024 GPSDyno Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Localization strings for GPSDyno.
Russian language dictionary for user interface.
"""

# Сообщения об ошибках
ERRORS = {
    'file_not_found': "Файл не найден: {file_path}",
    'insufficient_speed_data': "Недостаточно данных о скорости для расчета мощности (минимум {min_points} точек)",
    'calculation_failed': "Не удалось рассчитать мощность",
    'chart_failed': "Не удалось создать графики мощности",
    'only_nmea_supported': "Поддерживается только NMEA формат",
    'weather_server_error': "Сервер погоды не отвечает",
}

# Предупреждения (warnings) - критичные проблемы
WARNINGS = {
    'high_wind': "Ветер: скорость ветра превышает {threshold} км/ч ({speed:.1f} км/ч), что может искажать результаты",
    'low_gps_frequency': "Низкая частота обновления GPS: {freq:.1f} Гц (менее {threshold} Гц)",
    'high_filtered_ratio': "Отфильтровано {ratio:.1%} данных ({filtered} из {total} точек) из-за низкого качества GPS",
    'all_filtered': "Все {total} точек были отфильтрованы из-за низкого качества GPS",
    'unstable': "Данные ненадёжны: нестабильность {score:.0%}, погрешность увеличена",
    'gps_validity': "Потеряна точность GPS, возможны искажения",
    'gps_quality': "Серьёзные проблемы с качеством GPS: {ratio:.0%} точек с низким качеством сигнала",
    'log_issue': "Проблема с целостностью логов: обнаружено {count} пропусков в данных GPS",
}

# Предостережения (cautions) - менее критичные замечания
CAUTIONS = {
    'default_weather': "Используются погодные данные по умолчанию",
    'pre_kalman_filtered': "До фильтра Калмана исключено {ratio:.1%} точек ({count} шт.) из-за {reasons}",
    'wind': "Скорость ветра ({speed:.1f} км/ч) может влиять на точность результатов",
    'bad_weather': "Обнаружены неблагоприятные погодные условия: {conditions}",
    'gps_frequency': "Частота GPS ({freq:.1f} Гц) может привести к небольшим неточностям",
    'many_filtered': "Отфильтровано {ratio:.1%} данных ({filtered} из {total} точек) из-за низкого качества GPS",
    'unstable': "Обнаружена некоторая нестабильность данных (score: {score:.0%})",
    'gps_validity': "Наблюдаются периоды с невалидными данными GPS",
    'gps_quality': "Обнаружены признаки низкого качества GPS ({ratio:.0%} точек)",
    'log_issue': "Обнаружено {count} пропусков в данных GPS",
    'interpolated_data': "Данные интерполированы (не реальная частота GPS): {reasons}",
}

# Подписи осей и заголовки графиков
LABELS = {
    # Заголовки
    'power_by_speed_title': "Мощность по скорости на колесах (WHP)",
    'power_by_time_title': "Мощность по времени на колесах (WHP)",

    # Track visualization
    'track_power_label': "Мощность (л.с.)",

    # Оси
    'speed_axis': "Скорость (км/ч)",
    'power_axis': "Мощность (л.с.)",
    'time_axis': "Время (с)",

    # Легенды силовых составляющих
    'total_power': "Итоговая мощность (л.с.)",
    'air_resistance': "Мощность сопр. воздуха (л.с.)",
    'rolling_resistance': "Мощность сопр. качению (л.с.)",
    'slope_resistance': "Мощность сопр. уклону (л.с.)",
    'acceleration_power': "Мощность ускорения (л.с.)",
}

# Строки для погрешности (uncertainty)
UNCERTAINTY = {
    'total': "Погрешность",
    'wind': "Влияние ветра",
    'mass': "Погрешность массы",
    'gps': "Точность GPS",
    'consistency': "Множитель стабильности",
    'confidence_95': "95% доверительный интервал",
    'wind_not_counted': "ветер не учтён",
    'wind_headtail': "встречный/попутный {wind_kph:.0f} км/ч",
}
