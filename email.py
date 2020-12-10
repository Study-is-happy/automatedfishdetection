import imaplib

email = imaplib.IMAP4_SSL('imap.gmail.com')
email.login('isec.neufr@gmail.com', 'HSingh1!')

email.select('inbox')
_, email_id_list = email.search(
    None, '(' +
    'SUBJECT "[Amazon Mechanical Turk] Regarding Amazon Mechanical Turk HIT" ' +
    'FROM <mturk-noreply@amazon.com> ' +
    'UNANSWERED ' +
    'SINCE "07-Dec-2020"' +
    ')')

hit_id_list = []

for email_id in email_id_list[0].decode('utf-8').split():
    # BODY[TEXT]
    _, body = email.fetch(email_id, '(BODY.PEEK[TEXT])')

    body = body[0][1].decode('utf-8')
    for line in body.splitlines():
        if line.startswith('HIT ID:'):
            hit_id_list.append(line.split()[-1])
            break

email.close()
email.logout()

for results_path in os.listdir(config.project_dir + 'results/'):
    with open(results_approve_path) as results_approve_file:

        results = csv.reader(results_approve_file)

        next(results)
