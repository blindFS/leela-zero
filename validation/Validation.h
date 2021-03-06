#ifndef VALIDATION_H
#define VALIDATION_H
/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Marco Calignano

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <QTextStream>
#include <QString>
#include <QThread>
#include <QVector>
#include <QAtomicInt>
#include <QMutex>
#include "SPRT.h"
#include "../autogtp/Game.h"
#include "Results.h"

class ValidationWorker : public QThread {
    Q_OBJECT
public:
    enum {
        RUNNING = 0,
        FINISHING
    };
    ValidationWorker() = default;
    ValidationWorker(const ValidationWorker& w) : QThread(w.parent()) {}
    ~ValidationWorker() = default;
    void init(const QString& gpuIndex,
              const QString& firstNet,
              const QString& secondNet,
              const QString& keep,
              const QString& playOuts,
              int expected);
    void run() override;
    void doFinish() { m_state.store(FINISHING); }

signals:
    void resultReady(Sprt::GameResult r, int net_one_color);
private:
    QString m_firstNet;
    QString m_secondNet;
    int m_expected;
    QString m_keepPath;
    QString m_option;
    QAtomicInt m_state;
};

class Validation : public QObject {
    Q_OBJECT

public:
    Validation(const int gpus,
            const int games,
            const QString& playOuts,
            const float sigLevel,
            const QStringList& gpusList,
            const QString& firstNet,
            const QString& secondNet,
            const QString& keep,
            const QString& logFile,
            QMutex* mutex);
    ~Validation() = default;
    void startGames();
    void wait();

public slots:
    void getResult(Sprt::GameResult result, int net_one_color);

private:
    QMutex* m_mainMutex;
    QMutex m_syncMutex;
    Sprt m_statistic;
    Results m_results;
    QVector<ValidationWorker> m_gamesThreads;
    int m_games;
    int m_gpus;
    QString m_playOuts;
    float m_sigLevel;
    QStringList m_gpusList;
    QString m_firstNet;
    QString m_secondNet;
    QString m_keepPath;
    QString m_logFile;
    void writeLog(Sprt::Status status, std::tuple<int, int, int> wdl);
    QString fileHash(QString name);
    void quitThreads();
};

#endif
