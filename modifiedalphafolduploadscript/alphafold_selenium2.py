import time
import json
import argparse
import sys
import traceback
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options


# 初始化作业计数器
job_count = 0
MAX_JOBS = 20  # 每个目标最多提交20个作业后休眠24小时


def main():
    global job_count  # 访问全局变量
    parser = argparse.ArgumentParser(description='Automate AlphaFold server job submission.')
    parser.add_argument('--user_info', required=True, help='Path to user_info.json file')
    parser.add_argument('--sequences', required=True, help='Path to sequences.json file')
    parser.add_argument('--config', help='Path to config.json file with additional settings')
    parser.add_argument('--save', action='store_true', help='Save jobs after each sequence')
    args = parser.parse_args()

    # Load user info
    with open(args.user_info, 'r') as f:
        user_info = json.load(f)
    username_str = user_info.get('username')
    password_str = user_info.get('password')
    if not username_str or not password_str:
        print("Username or password not found in user_info.json")
        sys.exit(1)

    # Load sequences
    with open(args.sequences, 'r') as file:
        data = json.load(file)
    sequences = data.get('sequences', [])

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'copies': 2,
            'random_seed': 0  # Example setting, adjust as needed
        }

    # Initialize WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)")

    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    driver.get("https://alphafoldserver.com/")

    try:
        sign_in(driver)
        login(driver, username_str, password_str)
        accept_terms(driver)

        for i, sequence_data in enumerate(sequences):
            fasta_list = sequence_data.get('fasta', [])
            for j, fasta_entry in enumerate(fasta_list):
                # 获取当前 target 的所有链
                header, sequence = fasta_entry.split('\n', 1)
                process_sequence(driver, header, sequence, f"{i}_{j}", args.save, config)

            # 如果提交了 MAX_JOBS 次作业，则休眠24小时
            if job_count >= MAX_JOBS:
                print(f"已提交 {MAX_JOBS} 次作业，休眠 24 小时...")
                time.sleep(24 * 60 * 60)  # 休眠24小时
                job_count = 0  # 重置计数器，继续提交下一个目标的作业

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()
        driver.save_screenshot('error_screenshot.png')
        with open('error_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
    finally:
        time.sleep(10)
        driver.quit()


def sign_in(driver):
    '''sign_in_button = driver.find_element(By.CLASS_NAME,"sign-in")
    sign_in_button.click()'''
    wait = WebDriverWait(driver, random.randint(5, 10))
    sign_in_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "sign-in")))
    sign_in_button.click()


def login(driver, username_str, password_str):
    # Username
    username = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((By.ID, "identifierId"))
    )
    username.click()
    username.send_keys(username_str)
    username_next_button = driver.find_element(By.ID, "identifierNext")
    username_next_button.click()

    # Password
    password = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((By.NAME, "Passwd"))
    )
    password.click()
    password.send_keys(password_str)
    password_next_button = driver.find_element(By.ID, "passwordNext")
    password_next_button.click()


def accept_terms(driver):
    continue_button_found = False
    # Sometimes there will be terms of usage
    try:
        continue_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "continue"))
        )
        continue_button.click()
        continue_button_found = True

    except TimeoutException:
        pass

    if continue_button_found:
        # Initialize WebDriverWait
        wait = WebDriverWait(driver, 15)

        # Wait for the page to load completely
        wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')

        # Step 1: Find and click the checkbox
        try:
            checkbox_label = wait.until(EC.presence_of_element_located(
                (By.XPATH, "//label[contains(text(), 'I accept the')]")
            ))
            driver.execute_script("arguments[0].scrollIntoView(true);", checkbox_label)
            checkbox_id = checkbox_label.get_attribute('for')
            checkbox = driver.find_element(By.ID, checkbox_id)
            driver.execute_script("arguments[0].click();", checkbox)

        except Exception as e:
            pass

        # Step 2: Find and click the "Accept and continue" button
        try:
            accept_button_label = wait.until(EC.presence_of_element_located(
                (By.XPATH,
                 "//span[contains(@class, 'mdc-button__label') and normalize-space(text())='Accept and continue']")
            ))
            driver.execute_script("arguments[0].scrollIntoView(true);", accept_button_label)
            accept_button = accept_button_label.find_element(By.XPATH, "./ancestor::button")
            driver.execute_script("arguments[0].click();", accept_button)

        except Exception as e:
            pass

        # Step 3: Find and click the "Continue" button
        try:
            continue_button_label = wait.until(EC.presence_of_element_located(
                (By.XPATH, "//span[contains(@class, 'mdc-button__label') and normalize-space(text())='Continue']")
            ))
            driver.execute_script("arguments[0].scrollIntoView(true);", continue_button_label)
            continue_button = continue_button_label.find_element(By.XPATH, "./ancestor::button")
            driver.execute_script("arguments[0].click();", continue_button)

        except Exception as e:
            pass


def process_sequence(driver, header, sequence, index, save, config):
    copies_number = config.get('copies', 2)
    # First sequence: input data directly
    if index == 0:
        # Set number of copies
        copies = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@type='number' and @min='1']"))
        )
        copies.clear()
        copies.send_keys(str(copies_number))

        # Enter the sequence
        sequence_input = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.element_to_be_clickable((By.XPATH, "//textarea"))
        )
        sequence_input.clear()
        sequence_input.send_keys(sequence)

    # Other sequences: input new sequence and delete old, saved/processed sequence
    else:
        try:
            WebDriverWait(driver, random.randint(5, 10)).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, ".cdk-overlay-backdrop"))
            )
        except TimeoutException:
            print("Overlay did not disappear within the timeout period before adding new entity.")
            # Handle accordingly, maybe refresh the page or raise an exception
            driver.refresh()
            return

        # Add new protein
        add_button = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Add entity')]"))
        )
        add_button.click()

        # Set number of copies to copies_number on the newly added line
        copies_list = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.presence_of_all_elements_located((By.XPATH, "//input[@type='number' and @min='1']"))
        )
        copies = copies_list[-1]
        copies.clear()
        copies.send_keys(str(copies_number))

        # Enter the new sequence on the newly added line
        sequence_inputs = driver.find_elements(By.XPATH, "//textarea")
        sequence_input = sequence_inputs[-1]
        sequence_input.clear()
        sequence_input.send_keys(sequence)

        # Delete the processed / saved sequence
        menu_icon = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "div.menu button.mat-mdc-menu-trigger mat-icon.google-symbols"))
        )
        menu_icon.click()
        delete_button = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(.,'Delete') and contains(@class,'mat-mdc-menu-item')]"))
        )
        delete_button.click()

    if save:
        save_job(driver, header)
    else:
        submit_job(driver, header)


def save_job(driver, header):
    # Save job after entering the sequence
    save_job_button = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((
            By.XPATH, "//span[normalize-space(text())='Save job']/ancestor::button"
        ))
    )
    save_job_button.click()

    job_name = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((
            By.XPATH, "//input[contains(@class, 'mat-mdc-input-element') and @required and not(@type='number')]"
        ))
    )
    job_name.clear()
    sequence_id = header.split('|')[0].strip('>').split('_')[0]
    job_name.send_keys(sequence_id[:4])

    try:
        modal_save_button = WebDriverWait(driver, random.randint(5, 10)).until(
            EC.element_to_be_clickable((
                By.XPATH, "//button[contains(@class, 'confirm') and .//span[normalize-space(text())='Save job']]"
            ))
        )
        driver.execute_script("arguments[0].click();", modal_save_button)
    except Exception as e:
        print("Failed to click 'Save job' button:", e)
        driver.save_screenshot('error_screenshot.png')
        with open('error_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)


def submit_job(driver, header):
    # Preview then submit job directly
    continue_button = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Continue and preview job')]"))
    )
    continue_button.click()

    job_name = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((
            By.XPATH, "//input[contains(@class, 'mat-mdc-input-element') and @required and not(@type='number')]"
        ))
    )
    job_name.clear()
    sequence_id = header.split('|')[0].strip('>').split('_')[0]
    job_name.send_keys(sequence_id[:4])

    confirm_button = WebDriverWait(driver, random.randint(5, 10)).until(
        EC.element_to_be_clickable((
            By.XPATH, "//button[.//span[normalize-space(text())='Confirm and submit job']]"
        ))
    )
    confirm_button.click()


if __name__ == '__main__':
    main()

