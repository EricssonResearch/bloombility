from bloom import load_data


def main():
    num_clients = 10
    load_data.data_distributor.DATA_DISTRIBUTOR(num_clients)


if __name__ == "__main__":
    main()
