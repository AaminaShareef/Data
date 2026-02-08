# Auralis Insights  
### Data Visualization & Reporting Dashboard

Auralis Insights is a web-based data visualization and reporting dashboard designed to simplify data analysis through interactive dashboards, automated analytics, and report generation. The system enables users to securely upload datasets, preprocess data, visualize insights, and generate downloadable reports, reducing manual effort and improving decision-making efficiency.

---

## 📌 Features

- Secure user authentication with OTP-based email verification  
- User account creation and password recovery  
- Dataset upload (CSV / Excel)  
- Automated data preprocessing and cleaning  
- KPI calculation and analytics  
- Interactive data visualization dashboards  
- Automated report generation (PDF / Excel)  
- Report download and history tracking  

---

## 🏗️ System Architecture

The system follows a **three-tier architecture**:

- **Presentation Layer**  
  - Web interface using HTML, CSS, and JavaScript  

- **Application Layer**  
  - Backend built using Python (Django Framework)  
  - Handles authentication, data processing, analytics, and reporting  

- **Database Layer**  
  - SQLite database for storing users, datasets, analytics results, and reports  

---

## 🧩 Modules

1. **User Authentication Module**  
   - Login, signup, OTP verification  
   - Forgot password and reset functionality  

2. **Data Upload Module**  
   - Upload CSV and Excel files  
   - Data validation and preview  

3. **Data Preprocessing Module**  
   - Duplicate removal  
   - Missing value handling  
   - Data normalization  

4. **Analytics & KPI Module**  
   - Aggregation and metric computation  
   - Trend analysis  

5. **Visualization Dashboard**  
   - Interactive charts and graphs  
   - Filters and drill-down views  

6. **Report Generation Module**  
   - PDF and Excel report creation  
   - Report history and download  

---

## 🗃️ Database Design

Main entities:
- User  
- Dataset  
- Metrics  
- Report  

The database is designed to efficiently store uploaded data, computed analytics, and generated reports.

---

## ⚙️ Technologies Used

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python (Django Framework)  
- **Database:** SQLite  
- **Data Visualization:** Matplotlib, Plotly  
- **Authentication:** Email-based OTP verification  
- **Tools & IDE:** Visual Studio Code  
- **Version Control:** Git & GitHub  

---

## 🔄 Workflow

1. User registers or logs in  
2. OTP verification is completed  
3. User uploads dataset  
4. System preprocesses and cleans data  
5. Analytics and KPIs are generated  
6. Dashboard visualizes insights  
7. User generates and downloads reports  

---

## 🔐 Security Features

- Encrypted password storage  
- OTP-based email verification  
- Input validation  
- Secure session management  

---

## 📈 Future Enhancements

- Role-based access control  
- AI-based predictive analytics  
- Real-time data streaming  
- Mobile application support  
- Advanced visualization and alerts  

---

## 👩‍💻 Author

**Aamina Shareef**  
Integrated MCA Student  

---

## 📜 License

This project is developed for academic purposes.
