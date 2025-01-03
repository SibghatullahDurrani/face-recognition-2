import datetime
import sys
import urllib.parse
import uuid

import requests

employee_endpoint = "http://localhost:3000/employees/"
attendance_endpoint = "http://localhost:3000/attendance/"


def get_employee(id):
    url = employee_endpoint + str(id)
    return requests.get(url).json()


def get_employee_name(id):
    employee = get_employee(id)
    return employee["name"]


def manage_attendance(id):
    employee = get_employee(id)
    last_attendance_id = employee["lastAttendanceId"]
    last_attendance = get_attendance(last_attendance_id)
    last_attendance_time = last_attendance["time"]
    time_now = get_timestamp_now()
    if last_attendance_time + 30 < time_now:
        new_attendance_id = mark_attendance()
        patch_last_attendance_id(new_attendance_id, id)
        patch_attendance_ids(new_attendance_id, employee)
        print("attendance marked")


def get_attendance(id):
    url = attendance_endpoint + str(id)
    return requests.get(url).json()


def mark_attendance():
    url = attendance_endpoint
    data = {"id": get_uuid(), "time": get_timestamp_now()}
    requests.post(url, json=data)
    return data["id"]


def patch_last_attendance_id(attendance_id, employee_id):
    url = employee_endpoint + str(employee_id)
    data = {"lastAttendanceId": attendance_id}
    requests.patch(url, json=data)


def patch_attendance_ids(attendance_id, employee):
    url = employee_endpoint + str(employee["id"])
    existing_attendance_ids = employee["attendanceIds"]
    existing_attendance_ids.append(attendance_id)
    data = {"attendanceIds": existing_attendance_ids}
    requests.patch(url, json=data)


def get_timestamp_now():
    return int(datetime.datetime.now().timestamp())


def get_uuid():
    return str(uuid.uuid4())


def get_employee_by_name(name):
    url = employee_endpoint + "?"
    params = {"name": name}
    url = url + urllib.parse.urlencode(params)
    return requests.get(url).json()


def get_all_attendances_by_employee_name(name=None):
    if name is None:
        if len(sys.argv) == 2:
            name = sys.argv[1]
        else:
            return "Name is not provided"
    employee = get_employee_by_name(name)
    if len(employee) == 0:
        return "Employee not found"
    attendances = employee[0]["attendanceIds"]
    attendanceDates = []
    for attendance in attendances:
        attendanceTimeStamp = get_attendance(attendance)["time"]
        attendanceDates.append(
            str(datetime.datetime.fromtimestamp(attendanceTimeStamp))
        )
    return attendanceDates


print(get_all_attendances_by_employee_name())
