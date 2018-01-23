/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
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

#include <QtCore/QCoreApplication>
#include <QtCore/QTimer>
#include <QtCore/QTextStream>
#include <QtCore/QStringList>
#include <QCommandLineParser>
#include <QProcess>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <chrono>
#include <QCommandLineParser>
#include <iostream>
#include "../autogtp/Game.h"
#include "Validation.h"

constexpr int VALIDATION_VERSION = 1;

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("validation");
    app.setApplicationVersion(QString("v%1").arg(VALIDATION_VERSION));

    QTimer::singleShot(0, &app, SLOT(quit()));

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption networkOption(
        {"n", "network"},
            "Networks to use as players in competition mode (two are needed).",
            "filename");
    QCommandLineOption gamesNumOption(
        {"g", "gamesNum"},
            "Play 'gamesNum' games on one GPU at the same time.",
            "num", "1");
    QCommandLineOption playOutsOption(
        {"p", "playouts"},
            "Argument passed to leelaz (-p), defaults to 1600",
            "string", "1600");
    QCommandLineOption sigLevelOption(
        {"t", "siglevel"},
            "Significance level, defaults to 0.05",
            "num", "0.05");
    QCommandLineOption gpusOption(
        {"u", "gpus"},
            "Index of the GPU to use for multiple GPUs support.",
            "num");
    QCommandLineOption keepSgfOption(
        {"k", "keepSgf" },
            "Save SGF files after each self-play game.",
            "output directory");
    QCommandLineOption logOption(
        {"l", "logfile" },
            "Append results to which file.",
            "filename");

    parser.addOption(gamesNumOption);
    parser.addOption(gpusOption);
    parser.addOption(playOutsOption);
    parser.addOption(sigLevelOption);
    parser.addOption(networkOption);
    parser.addOption(keepSgfOption);
    parser.addOption(logOption);

    // Process the actual command line arguments given by the user
    parser.process(app);
    QStringList netList = parser.values(networkOption);
    if(netList.count() != 2) {
        parser.showHelp();
    }
    int gamesNum = parser.value(gamesNumOption).toInt();
    QString playOuts = parser.value(playOutsOption);
    float sigLevel = parser.value(sigLevelOption).toFloat();
    QStringList gpusList = parser.values(gpusOption);
    int gpusNum = gpusList.count();
    if (gpusNum == 0) {
        gpusNum = 1;
    }

    // Map streams
    QTextStream cout(stdout, QIODevice::WriteOnly);
    QTextStream cerr(stderr, QIODevice::WriteOnly);
    cerr << "validation v" << VALIDATION_VERSION << endl;
    if (parser.isSet(keepSgfOption)) {
        if (!QDir().mkpath(parser.value(keepSgfOption))) {
            cerr << "Couldn't create output directory for self-play SGF files!"
                 << endl;
            return EXIT_FAILURE;
        }
    }
    QMutex mutex;
    Validation validate(gpusNum, gamesNum,
                        playOuts, sigLevel, gpusList,
                        netList.at(0), netList.at(1),
                        parser.value(keepSgfOption),
                        parser.value(logOption), &mutex);
    validate.startGames();
    mutex.lock();
    cerr.flush();
    cout.flush();
    return app.exec();
}
